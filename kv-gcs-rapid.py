import time
import os
import json
import sys
import yaml
import gcsfs
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
GCS_BUCKET = "shiqi-rapid-west4-agent"
CONFIG_FILE = "lmcache_gcs_config.yaml"

# --- CACHE CONFIG ---
MAX_CPU_BUFFER_GB = 12.0

# --- MODEL CONFIG ---
MODEL_PATH = "Qwen/Qwen3-32B"
GPU_UTILIZATION = 0.95
MAX_MODEL_LEN = 32769
PROMPT_LEN = 32768

# --- GCSFS UTILS ---
fs = None

def get_fs():
    global fs
    if fs is None:
        fs = gcsfs.GCSFileSystem()
    return fs

def get_gcs_size_mb():
    total_size = 0
    file_count = 0
    try:
        # Query only the Qwen subfolder to avoid scanning the entire bucket
        res = get_fs().find(f"{GCS_BUCKET}/Qwen", detail=True)
        for f in res.values():
            total_size += f.get('size', 0)
            file_count += 1
    except FileNotFoundError:
        # Subfolder does not exist yet, which is normal before first write
        pass
    except Exception as e:
        print(f"   [GCS MONITOR] Error: {e}")
    return total_size / (1024 * 1024), file_count

def wait_for_gcs_stable(check_interval=2, stable_checks=3):
    print(f"\n[GCS MONITOR] Monitoring Native GCS Rapid bucket: {GCS_BUCKET}")
    
    last_size = -1
    stable_count = 0
    start_time = time.time()

    while True:
        current_size, count = get_gcs_size_mb()
        elapsed = time.time() - start_time

        speed_str = ""
        if last_size > 0:
            diff = current_size - last_size
            if diff > 0.01:
                speed_mbps = diff / check_interval
                speed_str = f"({speed_mbps:.2f} MB/s)"

        print(f"   T+{elapsed:.0f}s: {current_size:.2f} MB / {count} objects {speed_str}")

        if current_size == last_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_checks:
            print(f"[GCS MONITOR] Size stabilized.")
            break

        last_size = current_size
        time.sleep(check_interval)

    return time.time() - start_time

def cleanup_storage():
    print("\n" + "="*40)
    print(f"[CLEANUP] Wiping prefix 'Qwen' from GCS Rapid bucket...")
    print("="*40)
    import subprocess
    import sys
    try:
        # Run a separate Python subprocess using the exact same python executable to avoid gRPC pollution in the parent process
        cmd = [
            sys.executable,
            "-c",
            f"import gcsfs; fs = gcsfs.GCSFileSystem(); files = fs.find('{GCS_BUCKET}/Qwen'); fs.rm(files) if files else None"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print(f"   [CLEANUP] Successfully wiped files under gs://{GCS_BUCKET}/Qwen via gcsfs")
        else:
            print(f"   [CLEANUP] Warning: {res.stderr.strip()}")
    except Exception as e:
        print(f"   [CLEANUP] Warning: Failed to run cleanup: {e}")


# --- MAIN SCRIPT ---
def run_gcs_test():
    cleanup_storage()

    # Config using Native GCS connector
    config = {
        "chunk_size": 256,
        "local_cpu": False,
        "max_local_cpu_size": MAX_CPU_BUFFER_GB,
        "local_disk": None,
        "remote_url": f"gs://{GCS_BUCKET}",
        "remote_serde": "naive",
        "extra_config": {
            "gcs_max_workers": 128
        }
    }
    
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)

    os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath(CONFIG_FILE)
    
    # GCS doesn't use O_DIRECT (POSIX only)
    extra_config = {
        "gcs_max_workers": 128
    }
    os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)
    
    print(f"LMCache configuration written to {os.path.abspath(CONFIG_FILE)}")

    # Init Engine
    print(f"Initializing vLLM with LMCache (Native GCS Rapid Backend)...")
    kv_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        kv_transfer_config=kv_config,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=2048
    )

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0)

    # Payloads
    def make_prompt(id_char):
        suffix = tokenizer.encode(f" Question {id_char}", add_special_tokens=False)
        filler = tokenizer.encode(id_char * 5, add_special_tokens=False)[0]
        count = PROMPT_LEN - len(suffix)
        return ([filler] * count) + suffix

    prompt_A = make_prompt("A")
    prompt_B = make_prompt("B")

    print("\n" + "="*40)
    print("STARTING BENCHMARK: A -> B -> A")
    print("="*40)

    # ---------------------------------------------------------
    # Request 1: Content A (Compute + Write)
    # ---------------------------------------------------------
    print("\n[Request 1] Content A (Initial Prefill)")
    print("Expectation: Cache Miss. Compute + Write to GCS Rapid.")
    start = time.perf_counter()
    llm.generate(prompts=[{"prompt_token_ids": prompt_A}], sampling_params=params)
    dur_1 = time.perf_counter() - start
    print(f"-> Duration: {dur_1:.2f} s")
    
    wait_for_gcs_stable()

    # ---------------------------------------------------------
    # Request 2: Content B (Eviction)
    # ---------------------------------------------------------
    print("\n[Request 2] Content B (Forced Eviction)")
    print("Expectation: Cache Miss. Fills HBM, forcing A to be evicted from HBM.")
    start = time.perf_counter()
    llm.generate(prompts=[{"prompt_token_ids": prompt_B}], sampling_params=params)
    dur_2 = time.perf_counter() - start
    print(f"-> Duration: {dur_2:.2f} s")
    
    wait_for_gcs_stable()

    # ---------------------------------------------------------
    # Request 3: Content A (Retrieval)
    # ---------------------------------------------------------
    print("\n[Request 3] Content A (Retrieval)")
    print("Expectation: HBM Miss (evicted), LMCache Hit (Native GCS Read).")
    start = time.perf_counter()
    llm.generate(prompts=[{"prompt_token_ids": prompt_A}], sampling_params=params)
    dur_3 = time.perf_counter() - start
    print(f"-> Duration: {dur_3:.2f} s")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Req 1 (Compute A): {dur_1:.2f} s")
    print(f"Req 2 (Compute B): {dur_2:.2f} s")
    print(f"Req 3 (GCS Hit A): {dur_3:.2f} s")

if __name__ == "__main__":
    run_gcs_test()
