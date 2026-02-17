import sys
import os
import numpy as np
import time
import vulkan_nn_lib as torch
from vulkan_nn_lib.tensor import Tensor

def test_phase3_performance():
    print("--- Phase 3: ARAS Benchmark & Zero-Copy Test ---\n")
    
    # Setup SSD
    cache_dir = "/vectorlegis_ssd_pool/vnn_cache/test_p3"
    Tensor.setup_ssd_storage(cache_dir)
    
    # 1. Benchmark: Adaptive Streaming Addition
    # We'll use 4GB + 4GB. On a 24GB machine, this should trigger large tiles.
    size = 1024 * 1024 * 1024 # 1B elements = 4GB
    print(f"1. Benchmarking ARAS Addition (4GB + 4GB)...")
    
    a = torch.ones(size, device='ssd')
    b = torch.ones(size, device='ssd')
    
    t0 = time.perf_counter()
    res = a + b
    elapsed = time.perf_counter() - t0
    
    mb_s = (size * 4 * 3) / elapsed / 1e6 # A_read + B_read + Res_write
    print(f"   ARAS Addition done in {elapsed:.2f}s ({mb_s:.1f} MB/s total I/O)")
    
    # 2. Benchmark: Adaptive Matmul
    # Let's do a 8192 x 8192 matrix = 64M elements = 256MB. 
    # Small enough to fit in RAM for B-cache, but we want to see the budget logic.
    print(f"\n2. Benchmarking ARAS Matmul (8192 x 8192)...")
    m1 = torch.randn(8192, 8192, device='ssd')
    m2 = torch.randn(8192, 8192, device='ssd')
    
    t0 = time.perf_counter()
    m_res = m1 @ m2
    elapsed = time.perf_counter() - t0
    print(f"   ARAS Matmul done in {elapsed:.2f}s")
    
    # 3. Test: Zero-Copy Binary Mounting
    print(f"\n3. Testing Zero-Copy Binary Mounting...")
    # Create a dummy binary file
    bin_path = os.path.join(cache_dir, "dummy_weights.bin")
    dummy_data = np.arange(100, dtype=np.float32)
    dummy_data.tofile(bin_path)
    
    t0 = time.perf_counter()
    w = torch.from_binary(bin_path, shape=(10, 10))
    load_time = time.perf_counter() - t0
    
    print(f"   'Mounted' 100 elements in {load_time*1000:.4f}ms")
    sample = w.to_numpy().flatten()[:10]
    print(f"   Sample data: {sample}")
    
    assert np.all(sample == np.arange(10))
    print("   ✅ Zero-Copy Mounting OK")

    print("\n--- PHASE 3 PERFORMANCE TESTS COMPLETED ---")

if __name__ == "__main__":
    test_phase3_performance()
