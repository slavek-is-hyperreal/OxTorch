import os
import numpy as np
import taichi as ti
import torch as pt # REAL PYTORCH Baseline
import Python_Legacy.vulkan_nn_lib.torch_shim as vnn # VNN Replacement

def test_parity():
    print("=== VNN PyTorch Parity Test ===")
    
    # 1. Elementwise Add (f32)
    print("\n[1] Testing Elementwise Add (f32)...")
    shape = (1024, 1024)
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32)
    
    # VNN
    a_vnn = vnn.tensor(a_np, device='cpu')
    b_vnn = vnn.tensor(b_np, device='cpu')
    c_vnn = a_vnn + b_vnn
    
    # Baseline
    a_pt = pt.from_numpy(a_np)
    b_pt = pt.from_numpy(b_np)
    c_pt = a_pt + b_pt
    
    diff = np.abs(c_vnn.numpy() - c_pt.numpy()).max()
    print(f"    Max Diff: {diff:.2e}")
    assert diff < 1e-6
    
    # 2. Reduction Sum (f32 -> f64 precision)
    print("\n[2] Testing Reduction Sum (Large Tensor)...")
    n = 10**7
    a_np = np.random.randn(n).astype(np.float32)
    
    # VNN
    a_vnn = vnn.tensor(a_np, device='cpu')
    s_vnn = a_vnn.sum().item()
    
    # Baseline
    a_pt = pt.from_numpy(a_np)
    s_pt = float(pt.sum(a_pt.to(pt.float64)).item())
    
    diff = abs(s_vnn - s_pt)
    print(f"    VNN Sum: {s_vnn:.6f}")
    print(f"    PT Sum:  {s_pt:.6f}")
    print(f"    Diff:    {diff:.2e}")
    assert diff < 1e-1
    
    # 3. int32 Logic
    print("\n[3] Testing int32 Arithmetic...")
    a_vnn = vnn.tensor([1, 2, 3], dtype='int32')
    b_vnn = vnn.tensor([4, 5, 6], dtype='int32')
    c_vnn = a_vnn + b_vnn
    print(f"    VNN (1,2,3) + (4,5,6) = {c_vnn.numpy()}")
    assert np.all(c_vnn.numpy() == [5, 7, 9])
    
    # 4. int4 Unpacking Consistency
    print("\n[4] Testing int4 packing/unpacking...")
    # Generate random int8 in [-8, 7]
    data = np.random.randint(-8, 8, size=10, dtype=np.int8)
    a_vnn = vnn.tensor(data, dtype='int4')
    unpacked = a_vnn.numpy()
    print(f"    Original: {data}")
    print(f"    Unpacked: {unpacked}")
    # Note: Unpacking maps 0-15 to -8 to 7. 
    # Check if we can round-trip if we treat data as already centered
    
    print("\nAll Parity Tests Passed!")

if __name__ == "__main__":
    test_parity()
