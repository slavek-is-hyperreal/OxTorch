import sys
import os
import numpy as np

# Simulate user script:
import vulkan_nn_lib as torch
from vulkan_nn_lib.tensor import Tensor

def test_phase1():
    print("--- Phase 1: Compatibility & Multi-Dtype Test ---\n")
    
    # Setup SSD for large tests
    Tensor.setup_ssd_storage("/vectorlegis_ssd_pool/vnn_cache/test_compat")
    
    # 1. Basic Factory Functions
    print("1. Testing Factory Functions...")
    x = torch.randn(10, 10, dtype=torch.float32)
    y = torch.zeros(10, 10, dtype=torch.int32)
    z = torch.ones(5, 5, dtype=torch.int8)
    
    print(f"   x: {x.shape}, {x.dtype}, device={x.device}")
    print(f"   y: {y.shape}, {y.dtype}, device={y.device}")
    print(f"   z: {z.shape}, {z.dtype}, device={z.device}")
    
    assert x.dtype == np.float32
    assert y.dtype == np.int32
    assert z.dtype == np.int8
    assert np.all(z.to_numpy() == 1)
    print("   ✅ Factory functions OK")

    # 2. SSD Overflow with different dtypes
    print("\n2. Testing SSD Overflow (Multi-Dtype)...")
    # 1GB model on SSD (using float16 to save space if needed, or int8)
    # 512M elements int8 = 512MB
    # 512M elements float32 = 2GB
    size = 1024 * 1024 * 1024 # 1B elements
    
    print(f"   Creating 1B elements int8 tensor (1GB) on SSD...")
    big_int8 = torch.ones(size, dtype=torch.int8, device='ssd')
    print(f"   big_int8: {big_int8.shape}, {big_int8.dtype}, device={big_int8.device}")
    
    # Sample check
    sample = big_int8.to_numpy()[:10]
    print(f"   Sample data (int8): {sample}")
    assert big_int8.device == 'ssd'
    assert np.all(sample == 1)
    
    print(f"\n   Creating 512M elements float16 tensor (1GB) on SSD...")
    big_f16 = torch.zeros(512 * 1024 * 1024, dtype=torch.float16, device='ssd')
    print(f"   big_f16: {big_f16.shape}, {big_f16.dtype}, device={big_f16.device}")
    assert big_f16.device == 'ssd'
    assert big_f16.dtype == np.float16

    print("   ✅ SSD Multi-Dtype OK")

    # 3. Namespace compatibility
    print("\n3. Testing Namespace (torch.nn, torch.optim)...")
    m = torch.nn.Linear(10, 20)
    opt = torch.optim.SGD([m.weight], lr=1e-2)
    
    print(f"   Module: {type(m)}")
    print(f"   Optimizer: {type(opt)}")
    assert hasattr(torch, 'nn')
    assert hasattr(torch, 'optim')
    
    print("   ✅ Namespace compatibility OK")

    print("\n--- ALL PHASE 1 TESTS PASSED! ---")

if __name__ == "__main__":
    try:
        test_phase1()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
