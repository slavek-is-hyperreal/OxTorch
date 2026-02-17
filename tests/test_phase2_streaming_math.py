import sys
import os
import numpy as np
import time

# Simulate user script:
import vulkan_nn_lib as torch
from vulkan_nn_lib.tensor import Tensor

def test_phase2():
    print("--- Phase 2: Streaming Math (SOE) Test ---\n")
    
    # Setup SSD for large tests
    Tensor.setup_ssd_storage(None)
    
    # 1. 8GB Element-wise Addition (2B elements float32)
    # Total data involved: 8GB (A) + 8GB (B) + 8GB (Res) = 24GB on SSD.
    # This will definitely exceed RAM availability if loaded all at once.
    size = 1024 * 1024 * 1024 * 2 # 2B elements
    
    print(f"1. Testing Massive SSD Addition (2147M elements, 8GB)...")
    t0 = time.perf_counter()
    a = torch.ones(size, dtype=torch.float32, device='ssd') # All 1s
    b = torch.ones(size, dtype=torch.float32, device='ssd') # All 1s
    print(f"   Allocation time: {time.perf_counter()-t0:.2f}s")
    
    print("   Starting streaming addition (A + B)...")
    t0 = time.perf_counter()
    res = a + b
    elapsed = time.perf_counter() - t0
    print(f"   Addition completed in {elapsed:.2f}s ({size * 4 / elapsed / 1e6:.1f} MB/s)")
    
    # Verify samples
    sample_a = a.to_numpy()[:10]
    sample_b = b.to_numpy()[:10]
    sample_res = res.to_numpy()[:10]
    
    print(f"   Sample A: {sample_a}")
    print(f"   Sample B: {sample_b}")
    print(f"   Sample Res: {sample_res}")
    
    assert np.all(sample_res == 2.0)
    assert res.device == 'ssd'
    print("   ✅ Streaming Addition OK")

    # 2. SSD * Scalar
    print("\n2. Testing SSD * Scalar...")
    res_mul = a * 5.0
    sample_mul = res_mul.to_numpy()[-10:]
    print(f"   Sample Res (A*5): {sample_mul}")
    assert np.all(sample_mul == 5.0)
    print("   ✅ SSD Scalar Math OK")

    # 3. Tiled Matmul
    # 4000x4000 matrix = 16M elements = 64MB. Small for SSD but let's force it.
    print("\n3. Testing Tiled Matmul (SSD-resident)...")
    m1 = torch.ones(2000, 2000, device='ssd')
    m2 = torch.ones(2000, 2000, device='ssd')
    t0 = time.perf_counter()
    m_res = m1 @ m2
    print(f"   Matmul completed in {time.perf_counter()-t0:.2f}s")
    print(f"   Sample Res (corner): {m_res.to_numpy()[0,0]}")
    assert m_res.to_numpy()[0,0] == 2000.0
    print("   ✅ Tiled Matmul OK")

    print("\n--- ALL PHASE 2 TESTS PASSED! ---")

if __name__ == "__main__":
    try:
        test_phase2()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
