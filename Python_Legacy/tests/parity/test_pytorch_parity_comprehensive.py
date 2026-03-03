import torch
import numpy as np
import Python_Legacy.vulkan_nn_lib.torch_shim as vnn
from Python_Legacy.vulkan_nn_lib.memory import MemoryManager
from Python_Legacy.vulkan_nn_lib.tensor import Tensor
import Python_Legacy.vulkan_nn_lib.functional as F
import time
import os
import sys

# --- CONFIGURATION ---
DEBUG = False
ATOL_DEFAULT = 1e-4
# MOCK Budget for Hybrid/SSD testing
MemoryManager.get_safe_budget = lambda: 512 * 1024 * 1024 # 512MB
MemoryManager.should_tile = lambda size: size > 256 * 1024 * 1024 # Tile if > 256MB

def check_parity(name, v_t, t_t, atol=ATOL_DEFAULT):
    v_np = v_t.to_numpy()
    if isinstance(t_t, torch.Tensor):
        t_np = t_t.detach().cpu().numpy()
    else:
        t_np = np.array(t_t)
    
    # Reshape for scalar comparison if needed
    if v_np.shape != t_np.shape:
        if v_np.size == 1 and t_np.size == 1:
            v_np = v_np.item()
            t_np = t_np.item()
        else:
            return False, f"Shape mismatch: VNN {v_np.shape} != Torch {t_np.shape}"

    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, err_msg=f"Parity failure in {name}")
        return True, ""
    except AssertionError as e:
        return False, str(e)

def verify_arithmetic(mode, dtype_name):
    dtype = np.dtype(dtype_name)
    print(f"  [Unit] Arithmetic ({mode}, {dtype_name})...")
    N = 1000
    a_np = np.random.randint(1, 10, size=(N,)).astype(dtype)
    b_np = np.random.randint(1, 10, size=(N,)).astype(dtype)
    
    vt_a = vnn.tensor(a_np, device=mode)
    vt_b = vnn.tensor(b_np, device=mode)
    tt_a = torch.from_numpy(a_np)
    tt_b = torch.from_numpy(b_np)
    
    results = []
    # Add
    results.append(check_parity(f"ADD_{mode}_{dtype_name}", vt_a + vt_b, tt_a + tt_b))
    # Sub
    results.append(check_parity(f"SUB_{mode}_{dtype_name}", vt_a - vt_b, tt_a - tt_b))
    # Mul
    results.append(check_parity(f"MUL_{mode}_{dtype_name}", vt_a * vt_b, tt_a * tt_b))
    
    # Div (float result usually)
    if dtype_name != 'int4':
        v_res = vt_a / vt_b
        t_res = (tt_a.float() / tt_b.float())
        results.append(check_parity(f"DIV_{mode}_{dtype_name}", v_res, t_res))

    # Pow
    results.append(check_parity(f"POW_{mode}_{dtype_name}", vt_a ** 2, tt_a ** 2))
    
    return all(r[0] for r in results), [r[1] for r in results if not r[0]]

def verify_math_functions(mode, dtype_name):
    print(f"  [Unit] Math Functions ({mode}, {dtype_name})...")
    N = 1000
    # Use positive values for sqrt/log
    a_np = np.random.uniform(1.0, 10.0, size=(N,)).astype(np.float32)
    vt_a = vnn.tensor(a_np, device=mode)
    tt_a = torch.from_numpy(a_np)
    
    results = []
    # Sqrt
    results.append(check_parity(f"SQRT_{mode}", vt_a.sqrt(), tt_a.sqrt()))
    # Exp
    results.append(check_parity(f"EXP_{mode}", vt_a.exp(), tt_a.exp(), atol=1e-2))
    # Log
    results.append(check_parity(f"LOG_{mode}", vt_a.log(), tt_a.log()))
    # Tanh
    results.append(check_parity(f"TANH_{mode}", vt_a.tanh(), tt_a.tanh()))
    
    return all(r[0] for r in results), [r[1] for r in results if not r[0]]

def verify_activations(mode, dtype_name):
    print(f"  [Unit] Activations ({mode}, {dtype_name})...")
    N = 1000
    a_np = np.random.randn(N).astype(np.float32)
    vt_a = vnn.tensor(a_np, device=mode)
    tt_a = torch.from_numpy(a_np)
    
    results = []
    # Relu
    results.append(check_parity(f"RELU_{mode}", vt_a.relu(), torch.relu(tt_a)))
    # Leaky Relu
    results.append(check_parity(f"LEAKY_RELU_{mode}", vt_a.leaky_relu(alpha=0.1), torch.nn.functional.leaky_relu(tt_a, negative_slope=0.1)))
    # Silu
    results.append(check_parity(f"SILU_{mode}", vt_a.silu(), torch.nn.functional.silu(tt_a)))
    # Softmax
    a_2d = a_np.reshape(10, 100)
    vt_2d = vnn.tensor(a_2d, device=mode)
    tt_2d = torch.from_numpy(a_2d)
    results.append(check_parity(f"SOFTMAX_{mode}", vt_2d.softmax(dim=-1), torch.softmax(tt_2d, dim=-1), atol=1e-3))
    
    return all(r[0] for r in results), [r[1] for r in results if not r[0]]

def verify_structural(mode):
    print(f"  [Unit] Structural Ops ({mode})...")
    a_np = np.random.randn(4, 8, 16).astype(np.float32)
    vt_a = vnn.tensor(a_np, device=mode)
    tt_a = torch.from_numpy(a_np)
    
    results = []
    # Transpose
    results.append(check_parity(f"TRANSPOSE_{mode}", vt_a.transpose(0, 1), tt_a.transpose(0, 1)))
    # Permute
    results.append(check_parity(f"PERMUTE_{mode}", vt_a.permute(2, 0, 1), tt_a.permute(2, 0, 1)))
    # Flatten
    results.append(check_parity(f"FLATTEN_{mode}", vt_a.flatten(), tt_a.flatten()))
    # Reshape
    results.append(check_parity(f"RESHAPE_{mode}", vt_a.reshape(8, 64), tt_a.reshape(8, 64)))
    # Expand
    vt_b = vnn.tensor([1.0, 2.0], device=mode)
    tt_b = torch.tensor([1.0, 2.0])
    results.append(check_parity(f"EXPAND_{mode}", vt_b.expand(4, 2), tt_b.expand(4, 2)))
    
    return all(r[0] for r in results), [r[1] for r in results if not r[0]]

def verify_linear_algebra(mode):
    print(f"  [Unit] Linear Algebra ({mode})...")
    # 256x256 MatMul
    size = 256
    a_np = np.random.randn(size, size).astype(np.float32)
    b_np = np.random.randn(size, size).astype(np.float32)
    vt_a = vnn.tensor(a_np, device=mode)
    vt_b = vnn.tensor(b_np, device=mode)
    tt_a = torch.from_numpy(a_np)
    tt_b = torch.from_numpy(b_np)
    
    results = []
    # Matmul
    results.append(check_parity(f"MATMUL_{mode}", vt_a @ vt_b, tt_a @ tt_b, atol=1e-2))
    
    return all(r[0] for r in results), [r[1] for r in results if not r[0]]

def verify_integration(mode):
    print(f"\n[Integration] Simple MLP Chain ({mode})...")
    # Input -> Linear -> Relu -> Linear -> Sum
    X = np.random.randn(32, 128).astype(np.float32)
    W1 = np.random.randn(128, 64).astype(np.float32)
    W2 = np.random.randn(64, 10).astype(np.float32)
    
    vt_x = vnn.tensor(X, device=mode)
    vt_w1 = vnn.tensor(W1, device=mode)
    vt_w2 = vnn.tensor(W2, device=mode)
    
    def vnn_chain(x, w1, w2):
        h1 = (x @ w1).relu()
        out = (h1 @ w2).sum()
        return out
        
    tt_x = torch.from_numpy(X)
    tt_w1 = torch.from_numpy(W1)
    tt_w2 = torch.from_numpy(W2)
    
    def torch_chain(x, w1, w2):
        h1 = torch.relu(x @ w1)
        out = (h1 @ w2).sum()
        return out
        
    v_res = vnn_chain(vt_x, vt_w1, vt_w2)
    t_res = torch_chain(tt_x, tt_w1, tt_w2)
    
    ok, err = check_parity(f"INTEGRATION_MLP_{mode}", v_res, t_res, atol=1e-1)
    return ok, err

def run_benchmarks():
    print("\n" + "="*50)
    print("Performance Benchmarking (1.5x Requirement Check)")
    print("="*50)
    
    # 1. ADD Benchmark (8M elements ~32MB)
    size = 8 * 1024 * 1024
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)
    vt_a = vnn.tensor(a_np, device='cpu')
    vt_b = vnn.tensor(b_np, device='cpu')
    tt_a = torch.from_numpy(a_np)
    tt_b = torch.from_numpy(b_np)
    
    # Warmup
    _ = vt_a + vt_b
    _ = tt_a + tt_b
    
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        v_res = vt_a + vt_b
    t_vnn_add = (time.perf_counter() - t0) / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        t_res = tt_a + tt_b
    t_torch_add = (time.perf_counter() - t0) / iters
    
    ratio_add = t_vnn_add / t_torch_add
    print(f"Add (8M): VNN={t_vnn_add*1000:.1f}ms, Torch={t_torch_add*1000:.1f}ms | Slowdown: {ratio_add:.2f}x")
    
    # 2. Matmul Benchmark (1024x1024)
    size = 1024
    am_np = np.random.randn(size, size).astype(np.float32)
    bm_np = np.random.randn(size, size).astype(np.float32)
    vt_am = vnn.tensor(am_np, device='cpu')
    vt_bm = vnn.tensor(bm_np, device='cpu')
    tt_am = torch.from_numpy(am_np)
    tt_bm = torch.from_numpy(bm_np)
    
    t0 = time.perf_counter()
    for _ in range(5):
        v_res = vt_am @ vt_bm
    t_vnn_mm = (time.perf_counter() - t0) / 5
    
    t0 = time.perf_counter()
    for _ in range(5):
        t_res = tt_am @ tt_bm
    t_torch_mm = (time.perf_counter() - t0) / 5
    
    ratio_mm = t_vnn_mm / t_torch_mm
    print(f"MatMul (1024^2): VNN={t_vnn_mm*1000:.1f}ms, Torch={t_torch_mm*1000:.1f}ms | Slowdown: {ratio_mm:.2f}x")

    passed = (ratio_add <= 1.5) and (ratio_mm <= 1.5)
    if passed:
        print("✓ PERFORMANCE VERIFIED: All tests < 1.5x slowdown.")
    else:
        print("✗ PERFORMANCE WARNING: Some tests > 1.5x slowdown.")
    
    return passed

def run_all_tests():
    # Backends: cpu, vulkan, ssd, auto (hybrid)
    # We only test int dtypes for CPU/SSD primarily, as Vulkan is often f32-optimized
    modes = ['cpu', 'vulkan', 'ssd', 'auto']
    int_dtypes = ['int64', 'int32', 'int16', 'int8']
    
    print("="*60)
    print("COMPREHENSIVE PARITY SUITE: VNN vs PYTORCH")
    print("="*60)
    
    grand_results = []
    
    for mode in modes:
        print(f"\n--- TESTING MODE: {mode.upper()} ---")
        
        # Structural & Math functions (typically float32)
        ok, errors = verify_structural(mode)
        grand_results.append((f"STRUCTURAL_{mode}", ok, errors))
        
        ok, errors = verify_math_functions(mode, 'float32')
        grand_results.append((f"MATH_{mode}", ok, errors))
        
        ok, errors = verify_activations(mode, 'float32')
        grand_results.append((f"ACT_{mode}", ok, errors))
        
        ok, errors = verify_linear_algebra(mode)
        grand_results.append((f"LA_{mode}", ok, errors))

        # Integer Parity (Arithmetic)
        for dt in int_dtypes:
            ok, errors = verify_arithmetic(mode, dt)
            grand_results.append((f"ARITH_{mode}_{dt}", ok, errors))
            
        # Integration
        ok, err_msg = verify_integration(mode)
        grand_results.append((f"INTEGRATION_{mode}", ok, [err_msg] if not ok else []))

    # Benchmark
    perf_ok = run_benchmarks()
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL PARITY SUMMARY")
    print("="*60)
    passed_count = sum(1 for _, ok, _ in grand_results if ok)
    total_count = len(grand_results)
    
    print(f"Tests Passed: {passed_count}/{total_count}")
    
    if passed_count < total_count:
        print("\nTOP FAILURES:")
        for name, ok, errors in grand_results:
            if not ok:
                print(f"  [!] {name}: {errors[0] if errors else 'Unknown error'}")
        raise AssertionError("Parity tests failed.")
    else:
        print("\nALL PARITY TESTS PASSED! 100% Core Parity Achieved.")
        if not perf_ok:
            print("Note: Performance goal (1.5x) missed in some cases.")

def test_all_parity_cases():
    run_all_tests()

if __name__ == "__main__":
    run_all_tests()
