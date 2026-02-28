import os
import time
import numpy as np
import vulkannn_rusted as vnn

# 15 GB per tensor = 3.75 Billion f32 elements
# A: 15 GB, B: 15 GB, C: 15 GB (output)
# Total Operation footprint: 45 GB (completely crushes 24GB RAM limit)

N_ELEMENTS = 3_750_000_000
POOL_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(POOL_DIR, exist_ok=True)

FILE_A = os.path.join(POOL_DIR, "bench_15GB_A.bin")
FILE_B = os.path.join(POOL_DIR, "bench_15GB_B.bin")

def generate_random_file(filepath, num_elements):
    if os.path.exists(filepath):
        actual_size = os.path.getsize(filepath)
        expected_size = num_elements * 4
        if actual_size == expected_size:
            print(f"Skipping {filepath}, already exists ({actual_size / 1024**3:.2f} GB)")
            return
        else:
            print(f"File {filepath} exists but size mismatch, regenerating...")
            
    print(f"Generating {filepath} ({num_elements * 4 / 1024**3:.2f} GB) ...")
    
    chunk_size = 50_000_000 # ~200MB chunks
    chunks = num_elements // chunk_size
    remainder = num_elements % chunk_size
    
    with open(filepath, 'wb') as f:
        # We just generate ones to make it fast instead of `np.random.randn`
        # Math is math, and WGPU/CPU don't care about the values.
        chunk_data = np.ones(chunk_size, dtype=np.float32).tobytes()
        for i in range(chunks):
            f.write(chunk_data)
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{chunks}] chunks written...")
        
        if remainder > 0:
            f.write(np.ones(remainder, dtype=np.float32).tobytes())

def run_benchmark(mode="cpu"):
    # Drop caches to prevent OS cheating on the first run (Requires sudo, but 45GB I/O guarantees eviction anyway)
    # os.system('sync; echo 3 > /proc/sys/vm/drop_caches') 
    
    print(f"\n==============================================")
    print(f"[Benchmarking TRUE Rusted SSD Stream | Mode: {mode.upper()}]")
    print(f"==============================================")
    
    start_time = time.time()
    
    print("1. Mounting SSD Tensors (Zero-Copy mmap2)...")
    t1 = time.time()
    t_a = vnn.Tensor.from_ssd(FILE_A, [N_ELEMENTS])
    t_b = vnn.Tensor.from_ssd(FILE_B, [N_ELEMENTS])
    
    t_a.device = mode
    t_b.device = mode
    print(f"   Mount took: {time.time() - t1:.4f}s")
    
    print("2. Starting 45GB Math Execution (A + B = C)...")
    t2 = time.time()
    t_c = t_a + t_b
    assert t_c.shape == [N_ELEMENTS]
    print(f"   Execution took: {time.time() - t2:.4f}s")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Total I/O = Size A read + Size B read + Size C write
    total_gb = (N_ELEMENTS * 4 * 3) / 1024**3
    speed_mb_s = (total_gb * 1024) / duration
    
    print(f"\n>>> Mode: {mode.upper()} Final Results <<<")
    print(f"  -> Total Time : {duration:.4f} seconds")
    print(f"  -> I/O Bound  : ~{total_gb:.2f} GB processed")
    print(f"  -> Throughput : {speed_mb_s:.2f} MB/s")
    
    # To prevent Memory explosion when t_c (15GB rust vector) comes back into python scope,
    # we explicitly delete it and call gc.
    del t_c
    del t_a
    del t_b
    import gc
    gc.collect()
    
    return duration, speed_mb_s

def cleanup():
    # Only clean if you want to. For testing you might want to keep it.
    pass

if __name__ == "__main__":
    generate_random_file(FILE_A, N_ELEMENTS)
    generate_random_file(FILE_B, N_ELEMENTS)
    
    print("\nFiles ready. System RAM is ~24GB, Operation requires ~45GB I/O. Forcing Linux kernel streaming out-of-core!")
    
    # We test the Native Rusted fallbacks
    run_benchmark("cpu")
    run_benchmark("vulkan")
    run_benchmark("hybrid")
