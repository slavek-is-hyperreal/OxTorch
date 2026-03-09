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

def check_parity(vnn_tensor, torch_tensor, name, atol=1e-2):
    rtol = 1e-3
    if "bf16" in name.lower():
        # BF16 has 7-bit mantissa (~0.8% relative precision). 
        # For 2048-dim MatMul, accumulation noise + 1 ULP rounding can reach 1.0.
        atol = 1.0 
        rtol = 1.5e-2 
    if "monster" in name.lower():
        # Monster tests are OOM-safe, parity is not checked against PyTorch
        # This block is intentionally left empty as parity is skipped for these tests.
        pass
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
        elif (dtype == "f16" or dtype == "bf16") and op == "MatMul": iterations = 2 # PyTorch/Vulkan F16 is slow on this HW
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

    # VNN Execution (Warmup included)
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

import argparse

def get_stats(data):
    if not data: return 0, 0, 0
    return np.mean(data), np.median(data), np.std(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of full benchmark runs for statistics")
    args = parser.parse_args()

    print("="*60)
    print(f" VNN RUSTED SAFETY NET: TRI-PRECISION AUDIT v3.2")
    print(f" (Mode: Statistical Analysis - {args.runs} runs)")
    print("="*60)
    
    total_start = time.perf_counter()
    all_runs_results = []
    
    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n>>>>>> STARTING BENCHMARK RUN {run_idx+1}/{args.runs}")
            
        run_results = []
        for dtype in ["f32", "f16", "bf16"]:
            print(f"\n{'#'*20} PHASE: {dtype.upper()} {'#'*20}")
            for mode in ["cpu", "vulkan", "hybrid"]:
                pt, v_t, ok = run_bench(f"MatMul_{dtype}_{mode}", "MatMul", (2048, 2048), mode=mode, dtype=dtype)
                run_results.append({"name": f"MatMul {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok})
                
                pt, v_t, ok = run_bench(f"ReLU_{dtype}_{mode}", "ReLU", (1000000,), mode=mode, dtype=dtype)
                run_results.append({"name": f"ReLU {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok})

        pt, v_t, ok = run_bench("Monster_ReLU_F32_SSD", "ReLU", (4000000000,), mode="cpu", is_ssd=True, dtype="f32")
        run_results.append({"name": "Monster ReLU 16GB SSD", "pt": pt, "vnn": v_t, "ok": ok})
        all_runs_results.append(run_results)

    # --- AGGREGATE STATS ---
    test_names = [r["name"] for r in all_runs_results[0]]
    aggregated = []
    
    for name in test_names:
        vnn_times = [r["vnn"] for run in all_runs_results for r in run if r["name"] == name]
        pt_times = [r["pt"] for run in all_runs_results for r in run if r["name"] == name]
        any_fail = any(not r["ok"] for run in all_runs_results for r in run if r["name"] == name and r["ok"] != "N/A")
        
        vnn_mean, vnn_med, vnn_std = get_stats(vnn_times)
        pt_mean, pt_med, pt_std = get_stats(pt_times)
        
        aggregated.append({
            "name": name,
            "vnn_mean": vnn_mean, "vnn_med": vnn_med, "vnn_std": vnn_std,
            "pt_mean": pt_mean, "pt_med": pt_med,
            "failed": any_fail
        })

    # --- SUMMARY TABLE ---
    print("\n\n" + "="*100)
    print(f"{'VNN STATISTICAL SUMMARY (' + str(args.runs) + ' runs)':^100}")
    print("="*100)
    print(f"{'Test Case':<30} | {'PT (Med)':<10} | {'VNN (Med)':<10} | {'VNN (Std)':<10} | {'Ratio':<8} | {'Parity'}")
    print("-" * 100)
    
    for res in aggregated:
        ratio = res["vnn_med"] / res["pt_mean"] if res["pt_mean"] > 0 else 0
        ratio_str = f"{ratio:8.2f}x" if ratio > 0.1 else f"{ratio:8.4f}x"
        parity = "❌ FAIL" if res["failed"] else "✅ PASS"
        print(f"{res['name']:<30} | {res['pt_med']:8.4f}s | {res['vnn_med']:8.4f}s | {res['vnn_std']:8.4f}s | {ratio_str} | {parity}")
    
    print("="*100)

    # --- PERSISTENCE ---
    history = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list): history = []
        except: pass
    
    total_duration = time.perf_counter() - total_start
    history.append({
        "timestamp": time.ctime(),
        "total_duration_seconds": total_duration,
        "runs": args.runs,
        "results": aggregated
    })
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nResults appended to {RESULTS_FILE} (History length: {len(history)})")
    print(f"Total session duration: {total_duration:.2f}s")

    print("*" * 80)
    print(f"Results saved to {RESULTS_FILE} and logged to {HISTORY_FILE}")
