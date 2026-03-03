import torch
import numpy as np
import Python_Legacy.vulkan_nn_lib.torch_shim as vnn
import time
import os

def check_parity(name, v_t, t_t, atol=1e-5):
    v_np = v_t.to_numpy()
    if isinstance(t_t, torch.Tensor):
        t_np = t_t.detach().numpy()
    else:
        t_np = t_t
    
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, err_msg=f"Parity failure in {name}")
        return True, ""
    except AssertionError as e:
        return False, str(e)

def run_suite():
    modes = ['cpu', 'vulkan', 'ssd']
    dtypes = [np.int64, np.int32, np.int16, np.int8]
    
    # Results storage
    results = []

    print("="*50)
    print("VNN vs PyTorch Parity Test Suite")
    print("="*50)

    for mode in modes:
        for dtype in dtypes:
            dtype_name = np.dtype(dtype).name
            print(f"\nTesting Mode: {mode.upper()} | Dtype: {dtype_name}")
            
            # 1. Basic Operations
            # (Note: VNN often performs internal math in float32 for kernels,
            # so we test if the results are close after casting back if needed)
            
            a_np = np.random.randint(1, 10, size=(100, 100)).astype(dtype)
            b_np = np.random.randint(1, 10, size=(100, 100)).astype(dtype)
            
            vt_a = vnn.tensor(a_np, dtype=dtype, device=mode)
            vt_b = vnn.tensor(b_np, dtype=dtype, device=mode)
            
            tt_a = torch.from_numpy(a_np)
            tt_b = torch.from_numpy(b_np)
            
            # ADD
            v_res = vt_a + vt_b
            t_res = tt_a + tt_b
            ok, err = check_parity(f"ADD_{mode}_{dtype_name}", v_res, t_res)
            results.append((f"ADD_{mode}_{dtype_name}", ok, err))
            
            # SUB
            v_res = vt_a - vt_b
            t_res = tt_a - tt_b
            ok, err = check_parity(f"SUB_{mode}_{dtype_name}", v_res, t_res)
            results.append((f"SUB_{mode}_{dtype_name}", ok, err))

            # MUL
            v_res = vt_a * vt_b
            t_res = tt_a * tt_b
            ok, err = check_parity(f"MUL_{mode}_{dtype_name}", v_res, t_res)
            results.append((f"MUL_{mode}_{dtype_name}", ok, err))

            # SUM
            v_res = vt_a.sum()
            t_res = tt_a.sum()
            ok, err = check_parity(f"SUM_{mode}_{dtype_name}", v_res, t_res)
            results.append((f"SUM_{mode}_{dtype_name}", ok, err))

            # RELU (Functional)
            v_res = vt_a.relu()
            t_res = torch.relu(tt_a.float()).to(tt_a.dtype)
            ok, err = check_parity(f"RELU_{mode}_{dtype_name}", v_res, t_res)
            results.append((f"RELU_{mode}_{dtype_name}", ok, err))

    # Performance benchmark (CPU)
    print("\n" + "="*50)
    print("Performance Benchmarking (CPU Mode)")
    print("="*50)
    
    size = (4096, 4096) # Large enough to be meaningful
    a_np = np.random.randn(*size).astype(np.float32)
    b_np = np.random.randn(*size).astype(np.float32)
    
    vt_a = vnn.tensor(a_np, device='cpu')
    vt_b = vnn.tensor(b_np, device='cpu')
    tt_a = torch.from_numpy(a_np)
    tt_b = torch.from_numpy(b_np)
    
    # Warup
    _ = vt_a + vt_b
    _ = tt_a + tt_b
    
    t0 = time.perf_counter()
    for _ in range(10):
        v_res = vt_a + vt_b
    t_vnn = (time.perf_counter() - t0) / 10
    
    t0 = time.perf_counter()
    for _ in range(10):
        t_res = tt_a + tt_b
    t_torch = (time.perf_counter() - t0) / 10
    
    ratio = t_vnn / t_torch
    print(f"VNN Add: {t_vnn*1000:.2f}ms")
    print(f"Torch Add: {t_torch*1000:.2f}ms")
    print(f"Slowdown: {ratio:.2f}x")
    
    if ratio <= 1.5:
        print("✓ Performance requirement MET (<= 1.5x)")
    else:
        print("✗ Performance requirement FAILED (> 1.5x)")
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed < total:
        print("\nFailures:")
        for name, ok, err in results:
            if not ok:
                print(f"- {name}: {err[:100]}...")

if __name__ == "__main__":
    run_suite()
