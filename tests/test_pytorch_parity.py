import vulkan_nn_lib.torch_shim as torch
import numpy as np
import time
import os

def check(name, vnn_tensor, expected_np):
    vnn_np = vnn_tensor.to_numpy()
    if np.allclose(vnn_np, expected_np, atol=1e-5):
        print(f"  [PASS] {name}")
    else:
        err = np.abs(vnn_np - expected_np).max()
        print(f"  [FAIL] {name} (Max Err: {err:.6f})")

def run_parity_test(scale_name, size):
    print(f"\n--- Testing Scale: {scale_name} ({size} elements) ---")
    
    # 1. Slicing & Indexing
    x = torch.randn(size, 10, device='auto')
    expected_x = x.to_numpy()
    
    check("indexing x[idx]", x[size//2], expected_x[size//2])
    check("slicing x[0:5]", x[0:5], expected_x[0:5])
    
    # 2. Join/Split
    y = torch.ones(size, 10, device='auto')
    expected_y = np.ones((size, 10), dtype=np.float32)
    
    z = torch.cat([x, y], dim=0)
    expected_z = np.concatenate([expected_x, expected_y], axis=0)
    check("cat(dim=0)", z, expected_z)
    
    # 3. Shape Ops
    x_p = x.permute(1, 0)
    expected_xp = expected_x.transpose()
    check("permute(1, 0)", x_p, expected_xp)
    
    # 4. Advanced Math
    mask = x > 0
    xf = x.masked_fill(mask, -1.0)
    expected_xf = expected_x.copy()
    expected_xf[expected_x > 0] = -1.0
    check("masked_fill", xf, expected_xf)
    
    x_exp = x.exp()
    check("exp()", x_exp, np.exp(expected_x))

if __name__ == "__main__":
    print("VNN API Parity Verification Suite")
    
    # scale tiny
    run_parity_test("Tiny (RAM)", 100)
    
    # scale large
    run_parity_test("Large (RAM/SSD Hybrid)", 100000)
    
    # Monster Scale (If possible, check if SSD is used)
    # 100M elements * 10 * 4 bytes = 4GB per tensor
    # We test with 250M total to see streaming in action
    try:
        run_parity_test("Monster (SSD-Native)", 25000000)
    except Exception as e:
        print(f"\n[!] Monster Scale error: {e}")
