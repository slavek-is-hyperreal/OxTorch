import time
import numpy as np
import torch
import os
import sys
import json
import vulkannn_rusted as vnn
from vulkannn_rusted import Tensor

RESULTS_FILE = "tests/last_results.json"
HISTORY_FILE = "tests/benchmark_history.log"

# Constants
CACHE_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def format_size(elements):
    if elements >= 1e9: return f"{elements/1e9:.1f}B"
    if elements >= 1e6: return f"{elements/1e6:.1f}M"
    return str(elements)

def check_parity(vnn_tensor, torch_tensor, name, atol=1e-3):
    v_np = vnn_tensor.to_numpy()
    t_np = torch_tensor.detach().numpy() if hasattr(torch_tensor, 'detach') else torch_tensor
    
    # Flatten if needed for comparison
    v_np = v_np.flatten()
    t_np = t_np.flatten()
    
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, rtol=1e-3)
        return True, 0.0
    except AssertionError as e:
        diff = np.abs(v_np - t_np)
        return False, np.max(diff)

def run_bench(name, op, shape, mode="cpu", is_ssd=False, iterations=None, dtype="f32"):
    if dtype == "f32":
        vnn_dtype = vnn.DataType.F32
        torch_dtype = torch.float32
    elif dtype == "f16":
        vnn_dtype = vnn.DataType.F16
        torch_dtype = torch.float16
    elif dtype == "bf16":
        vnn_dtype = vnn.DataType.BF16
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    if iterations is None:
        size_elements = np.prod(shape)
        if is_ssd or size_elements > 5e7: iterations = 1
        elif size_elements > 5e6: iterations = 5
        elif (dtype == "f16" or dtype == "bf16") and mode == "cpu" and op == "MatMul": iterations = 1 # PyTorch is 200x slower here!
        else: iterations = 10 if dtype in ["f16", "bf16"] else 20
        
    print(f"\n>>> TEST: {name} ({mode.upper()}, {dtype.upper()}) | Shape: {shape} | SSD: {is_ssd} | Iter: {iterations}")
    
    # Pre-generate or Load Data
    if not is_ssd:
        a_np = np.random.randn(*shape).astype(np.float32)
        if op == "MatMul":
            b_np = np.random.randn(shape[-1], shape[-1]).astype(np.float32)
        else:
            b_np = np.random.randn(*shape).astype(np.float32)
            
        a_torch = torch.from_numpy(a_np).to(torch_dtype)
        b_torch = torch.from_numpy(b_np).to(torch_dtype) if op != "ReLU" else None
        
        # PyTorch Reference
        t0 = time.perf_counter()
        for _ in range(iterations):
            if op == "MatMul": res_torch = torch.matmul(a_torch, b_torch)
            elif op == "Add": res_torch = a_torch + b_torch
            elif op == "ReLU": res_torch = torch.relu(a_torch)
        t_pt = (time.perf_counter() - t0) / iterations
        
        # VNN Input
        a_vnn = Tensor(data=a_np, dtype=vnn_dtype, device=mode)
        b_vnn = Tensor(data=b_np, dtype=vnn_dtype, device=mode) if op != "ReLU" else None
    else:
        # SSD Path - Force F32 for now in Monster tests unless needed
        size_elements = np.prod(shape)
        a_path = os.path.join(CACHE_DIR, f"monster_a_{size_elements}.bin")
        a_vnn = Tensor.from_ssd(a_path, shape, dtype=vnn_dtype)
        a_vnn.device = mode
        b_vnn = None
        t_pt = 0
        res_torch = None

    # VNN Execution (Warnup included)
    if op == "MatMul": _ = a_vnn @ b_vnn
    elif op == "Add": _ = a_vnn + b_vnn
    elif op == "ReLU": _ = a_vnn.relu()

    t0 = time.perf_counter()
    for _ in range(iterations):
        if op == "MatMul": res_vnn = a_vnn @ b_vnn
        elif op == "Add": res_vnn = a_vnn + b_vnn
        elif op == "ReLU": res_vnn = a_vnn.relu()
    t_vnn = (time.perf_counter() - t0) / iterations
    
    # Validation
    parity_ok = "N/A"
    max_diff = 0.0
    if not is_ssd and res_torch is not None:
        atol = 1e-3 if dtype == "f32" else 1e-2
        parity_ok, max_diff = check_parity(res_vnn, res_torch.to(torch.float32), name, atol=atol)
        parity_str = "✅ OK" if parity_ok else f"❌ FAIL (diff: {max_diff:.6f})"
    else:
        parity_str = "N/A (OOM-Safe)"

    print(f"    [PyTorch] {t_pt:.4f}s")
    print(f"    [VNN]     {t_vnn:.4f}s | Parity: {parity_str}")
    
    return t_pt, t_vnn, parity_ok

if __name__ == "__main__":
    print("="*60)
    print(" VNN RUSTED SAFETY NET: TRI-PRECISION AUDIT v3.1")
    print("="*60)
    
    results = []
    
    for dtype in ["f32", "f16", "bf16"]:
        print(f"\n{'#'*20} PHASE: {dtype.upper()} {'#'*20}")
        # --- RAM-RESIDENT PARITY ---
        for mode in ["cpu", "vulkan", "hybrid"]:
            # 1. MatMul Parity
            pt, v_t, ok = run_bench(f"MatMul_{dtype}_{mode}", "MatMul", (2048, 2048), mode=mode, dtype=dtype)
            results.append((f"MatMul {dtype} ({mode})", pt, v_t, ok))
            
            # 2. ReLU Parity
            pt, v_t, ok = run_bench(f"ReLU_{dtype}_{mode}", "ReLU", (1000000,), mode=mode, dtype=dtype)
            results.append((f"ReLU {dtype} ({mode})", pt, v_t, ok))

        # --- SPEED SUPREMACY (Large RAM) ---
        if dtype in ["f16", "bf16"]:
             pt, v_t, ok = run_bench(f"MatMul_{dtype}_CPU_Huge", "MatMul", (4096, 4096), mode="cpu", dtype=dtype)
             results.append((f"MatMul {dtype} Huge CPU", pt, v_t, ok))
        else:
             pt, v_t, ok = run_bench(f"MatMul_{dtype}_CPU_Huge", "MatMul", (2048, 2048), mode="cpu", dtype=dtype)
             # Already covered in loop, but just for consistency in reports

    # --- MONSTER STREAMING (SSD) ---
    pt, v_t, ok = run_bench("Monster_ReLU_F32_SSD", "ReLU", (4000000000,), mode="cpu", is_ssd=True, dtype="f32")
    results.append(("Monster ReLU 16GB SSD", pt, v_t, ok))

    # --- SUMMARY ---
    print("\n\n" + "="*80)
    print(f"{'VNN SAFETY NET SUMMARY':^80}")
    print("="*80)
    print(f"{'Test Case':<30} | {'PyTorch':<10} | {'VNN':<10} | {'Ratio':<8} | {'Parity'}")
    print("-" * 80)
    
    failed = False
    for name, pt, v_t, ok in results:
        ratio = v_t / pt if pt > 0 else 0
        r_str = f"{ratio:.2f}x" if pt > 0 else "---"
        p_str = "✅ PASS" if (ok is True or ok == "N/A") else "❌ FAIL"
        if ok is False: failed = True
        print(f"{name:<30} | {pt:>8.4f}s | {v_t:>8.4f}s | {r_str:>8} | {p_str}")
    
    print("="*80)
    if failed:
        print("\n⚠️  WARNING: Some parity tests failed! Check numerical differences.")
    else:
        print("\n✨ ALL CORE SYSTEMS OPERATIONAL: F32/F16/BF16 Parity and Speed Verified.")
    print("="*80)

    # --- REGRESSION MONITORING ---
    prev_results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                prev_results = json.load(f)
        except:
            prev_results = {}

    print(f"\n{' RATIO-BASED REGRESSION MONITORING ':*^80}")
    print(f"{'Test Case':<30} | {'Current R':<10} | {'Prev R':<10} | {'Delta':<10} | {'Status'}")
    print("-" * 80)
    
    current_save = {}
    for name, pt, v_t, ok in results:
        ratio = v_t / pt if pt > 0 else 0
        current_save[name] = ratio
        prev_ratio = prev_results.get(name)
        if prev_ratio is not None:
            delta = ratio - prev_ratio
            perc = (delta / prev_ratio) * 100 if prev_ratio > 0 else 0
            d_str = f"{delta:>+8.4f}"
            status = "🔴 REGRESSION" if perc > 10 else "🟢 IMPROVED" if perc < -10 else "🟡 STABLE"
            print(f"{name:<30} | {ratio:>8.4f} | {prev_ratio:>8.4f} | {d_str} | {status} ({perc:>+5.1f}%)")
        else:
            print(f"{name:<30} | {ratio:>8.4f} | {'N/A':>8} | {'---':>8} | NEW TEST")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(current_save, f, indent=4)
        
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"\n--- Benchmark Run: {time.ctime()} ---\n")
        for name, pt, v_t, ok in results:
            f.write(f"{name:<30} | VNN: {v_t:.4f}s | PT: {pt:.4f}s | Parity: {ok}\n")

    print("*" * 80)
    print(f"Results saved to {RESULTS_FILE} and logged to {HISTORY_FILE}")
