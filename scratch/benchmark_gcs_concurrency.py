import time
import asyncio
import gcsfs
import os
import sys
from concurrent.futures import ThreadPoolExecutor

GCS_BUCKET = "shiqi-rapid-west4-agent"
CHUNK_SIZE_MB = 64
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024
NUM_CHUNKS = 8  # Test with 8 chunks (512 MB total) to avoid massive bucket bloat during quick runs

# Pre-allocate dummy chunk buffer
dummy_data = bytearray(CHUNK_SIZE_BYTES)

def cleanup_benchmark_gcs():
    fs = gcsfs.GCSFileSystem(consistency='none')
    try:
        files = fs.find(f"{GCS_BUCKET}/benchmark_test")
        if files:
            fs.rm(files)
    except Exception:
        pass

def test_sync_open_write(fs, path, buffer):
    with fs.open(path, 'wb', block_size=CHUNK_SIZE_BYTES) as f:
        f.write(buffer)

def test_sync_pipe_file(fs, path, buffer):
    fs.pipe_file(path, buffer)

def test_sync_readinto(fs, path, buffer):
    with fs.open(path, 'rb', block_size=CHUNK_SIZE_BYTES) as f:
        return f.readinto(buffer)

def run_parallel_test(method_fn, fs, num_workers, use_pipe=False):
    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures = []
    
    start_time = time.perf_counter()
    for i in range(NUM_CHUNKS):
        path = f"{GCS_BUCKET}/benchmark_test/chunk_{i}.data"
        # Submit to thread pool
        fut = executor.submit(method_fn, fs, path, dummy_data)
        futures.append(fut)
        
    # Wait for all to complete
    for fut in futures:
        fut.result()
        
    duration = time.perf_counter() - start_time
    executor.shutdown(wait=True)
    return duration

def main():
    print("="*60)
    print(f"GCS RAPID CONCURRENCY BENCHMARK (Bucket: {GCS_BUCKET})")
    print(f"Data: {NUM_CHUNKS} chunks of {CHUNK_SIZE_MB} MB each = {NUM_CHUNKS * CHUNK_SIZE_MB} MB total")
    print("="*60)
    
    fs = gcsfs.GCSFileSystem(consistency='none')
    cleanup_benchmark_gcs()
    
    results = []
    
    # Test different concurrency limits
    concurrency_levels = [1, 4, 8, 16]
    
    for workers in concurrency_levels:
        print(f"\n--- Running with {workers} worker thread(s) ---")
        
        # 1. Test Upload via standard open/write
        cleanup_benchmark_gcs()
        dur_write = run_parallel_test(test_sync_open_write, fs, workers)
        tput_write = (NUM_CHUNKS * CHUNK_SIZE_MB) / dur_write
        print(f"   [Upload: open/write] Duration: {dur_write:.2f}s, Throughput: {tput_write:.2f} MB/s")
        
        # 2. Test Upload via pipe_file
        cleanup_benchmark_gcs()
        dur_pipe = run_parallel_test(test_sync_pipe_file, fs, workers)
        tput_pipe = (NUM_CHUNKS * CHUNK_SIZE_MB) / dur_pipe
        print(f"   [Upload: pipe_file]  Duration: {dur_pipe:.2f}s, Throughput: {tput_pipe:.2f} MB/s")
        
        # 3. Test Download via readinto (uses the files uploaded by pipe_file)
        dur_read = run_parallel_test(test_sync_readinto, fs, workers)
        tput_read = (NUM_CHUNKS * CHUNK_SIZE_MB) / dur_read
        print(f"   [Download: readinto] Duration: {dur_read:.2f}s, Throughput: {tput_read:.2f} MB/s")
        
        results.append({
            "workers": workers,
            "write_dur": dur_write,
            "write_tput": tput_write,
            "pipe_dur": dur_pipe,
            "pipe_tput": tput_pipe,
            "read_dur": dur_read,
            "read_tput": tput_read
        })
        
    # Print Markdown Table
    print("\n" + "="*60)
    print("BENCHMARK RESULT SUMMARY")
    print("="*60)
    print("| Workers | open/write Throughput | pipe_file Throughput | readinto Throughput |")
    print("|---------|-----------------------|----------------------|---------------------|")
    for r in results:
        print(f"| {r['workers']:7d} | {r['write_tput']:19.2f} MB/s | {r['pipe_tput']:18.2f} MB/s | {r['read_tput']:17.2f} MB/s |")
    print("="*60)
    
    cleanup_benchmark_gcs()

if __name__ == "__main__":
    main()
