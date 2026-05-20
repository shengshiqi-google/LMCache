import time
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
MODEL_PATH = "Qwen/Qwen3-32B"
GPU_UTILIZATION = 0.95
MAX_MODEL_LEN = 32769
PROMPT_LEN = 32768

print("Initializing baseline vLLM (without LMCache)...")
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=GPU_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    enable_prefix_caching=True,
    enforce_eager=False,
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_num_batched_tokens=2048
)

tokenizer = llm.get_tokenizer()
params = SamplingParams(max_tokens=1, temperature=0)

# Payload
def make_prompt(id_char):
    suffix = tokenizer.encode(f" Question {id_char}", add_special_tokens=False)
    filler = tokenizer.encode(id_char * 5, add_special_tokens=False)[0]
    count = PROMPT_LEN - len(suffix)
    return ([filler] * count) + suffix

prompt_A = make_prompt("A")

print("\nStarting baseline prefill measurement...")
start = time.perf_counter()
llm.generate(prompts=[{"prompt_token_ids": prompt_A}], sampling_params=params)
dur = time.perf_counter() - start
print(f"-> Baseline Prefill Duration: {dur:.2f} s")
