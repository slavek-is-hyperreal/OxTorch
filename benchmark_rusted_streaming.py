import os
import time
import numpy as np
import pytest
import vulkannn_rusted as vnn

# Create huge dummy files to force SSD I/O. 
# We'll use 50 Million elements per tensor (~200MB each) so the total 
# operation (A + B = C) takes 600MB of I/O.
# We then run it multiple times to ensure the bandwidth is tested.

N_ELEMENTS = 50_000_000 
FILE_A = "bench_A.bin"
FILE_B = "bench_B.bin"

def setup_files():
    if not os.path.exists(FILE_A):
        print(f"Generating {FILE_A}... ({N_ELEMENTS * 4 / 1024 / 1024:.1f} MB)")
        # chunked writing to not explode Python RAM
        with open(FILE_A, 'wb') as f:
            for _ in range(50):
                f.write(np.ones(N_ELEMENTS // 50, dtype=np.float32).tobytes())
                
    if not os.path.exists(FILE_B):
        print(f"Generating {FILE_B}... ({N_ELEMENTS * 4 / 1024 / 1024:.1f} MB)")
        with open(FILE_B, 'wb') as f:
            for _ in range(50):
                f.write((np.ones(N_ELEMENTS // 50, dtype=np.float32) * 2.0).tobytes())

def run_benchmark(mode="cpu"):
    # Clear OS cache if possible (requires root on Linux, bypassing for now, 
    # but the first run might be slightly slower due to cold disk).
    
    print(f"\n[Benchmarking Rusted Engine | Mode: {mode.upper()}]")
    start_time = time.time()
    
    # 1. Zero-Copy SSD Mounts
    t_a = vnn.Tensor.from_ssd(FILE_A, [N_ELEMENTS])
    t_b = vnn.Tensor.from_ssd(FILE_B, [N_ELEMENTS])
    
    # Set the mathematical target device
    t_a.device = mode
    t_b.device = mode
    
    # 2. Add (Reads 400MB from SSD, Writes 200MB to RAM/GPU)
    t_c = t_a + t_b
    
    # Force GPU Sync if it's WGPU (Rust backend blocks automatically, but just checking)
    assert t_c.shape == [N_ELEMENTS]
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Total I/O = Size A + Size B (Reads) + Size C (Writes to RAM returned by Rust)
    total_mb = (N_ELEMENTS * 4 * 3) / 1024 / 1024
    speed_mb_s = total_mb / duration
    
    print(f"  -> Time      : {duration:.4f} seconds")
    print(f"  -> Throughput: {speed_mb_s:.2f} MB/s")
    
    return duration, speed_mb_s

def cleanup():
    if os.path.exists(FILE_A): os.remove(FILE_A)
    if os.path.exists(FILE_B): os.remove(FILE_B)

if __name__ == "__main__":
    setup_files()
    
    try:
        # We test the Native Rusted fallbacks
        run_benchmark("cpu")
        
        # WGPU natively in Rust
        run_benchmark("vulkan") # We'll route this to native WGPU
        
        # True Heterogeneous (Rayon + WGPU Ping-Pong)
        run_benchmark("hybrid")
        
    finally:
        print("\nCleaning up large binary files...")
        cleanup()
