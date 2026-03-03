import os
import time
import numpy as np
import vulkannn_rusted as vnn

# MatMul Benchmark
# M=10000, K=10000, N=10000 -> Tensors A: 400MB, B: 400MB, C: 400MB (1.2GB total)
# 400MB is small enough to finish in a decent time, but large enough 
# to trigger block-tiling if VRAM or block_size is restricted artificially 
# (our block_size is 2048x2048 = 16MB)

M = 10000
K = 10000
N = 10000

POOL_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(POOL_DIR, exist_ok=True)

FILE_A = os.path.join(POOL_DIR, "bench_matmul_A.bin")
FILE_B = os.path.join(POOL_DIR, "bench_matmul_B.bin")

def generate_random_file(filepath, num_elements):
    if os.path.exists(filepath):
        actual_size = os.path.getsize(filepath)
        expected_size = num_elements * 4
        if actual_size == expected_size:
            print(f"Skipping {filepath}, already exists ({actual_size / 1024**2:.2f} MB)")
            return
            
    print(f"Generating {filepath} ({num_elements * 4 / 1024**2:.2f} MB) ...")
    with open(filepath, 'wb') as f:
        # Generate sequentially for speed
        chunk_data = np.random.rand(1_000_000).astype(np.float32).tobytes()
        written = 0
        while written < num_elements:
            to_write = min(1_000_000, num_elements - written)
            if to_write == 1_000_000:
                f.write(chunk_data)
            else:
                f.write(np.random.rand(to_write).astype(np.float32).tobytes())
            written += to_write

def run_benchmark(mode="cpu"):
    print(f"\n==============================================")
    print(f"[Benchmarking Tiled MatMul SSD | Mode: {mode.upper()}]")
    print(f"==============================================")
    
    start_time = time.time()
    
    t_a = vnn.Tensor.from_ssd(FILE_A, [M, K])
    t_b = vnn.Tensor.from_ssd(FILE_B, [K, N])
    
    t_a.device = mode
    t_b.device = mode
    
    t1 = time.time()
    t_c = t_a @ t_b
    assert t_c.shape == [M, N]
    print(f"   Execution took: {time.time() - t1:.4f}s")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # GFLOPs = 2 * M * N * K / 1e9
    gflops = (2.0 * M * N * K) / 1e9
    tflops_s = (gflops / 1e3) / duration
    
    print(f"\n>>> Mode: {mode.upper()} Final Results <<<")
    print(f"  -> Total Time : {duration:.4f} seconds")
    print(f"  -> Compute    : {tflops_s:.4f} TFLOPS")

    del t_c
    del t_a
    del t_b
    import gc
    gc.collect()

if __name__ == "__main__":
    generate_random_file(FILE_A, M * K)
    generate_random_file(FILE_B, K * N)
    
    print("\nFiles ready. Testing Out-Of-Core Tiled MatMul Engine!")
    
    run_benchmark("cpu")
    run_benchmark("hybrid")
    run_benchmark("vulkan")
