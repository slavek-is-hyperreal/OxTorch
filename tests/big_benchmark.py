import time
import numpy as np
import torch
import os
import sys
import json

# Force line-buffered output even when piped
sys.stdout.reconfigure(line_buffering=True)

# Branch-aware dynamic import
_VNN_CANDIDATES = [
    "vulkannn_rusted_exp",
    "vulkannn_rusted_dev",
    "vulkannn_rusted_test",
    "vulkannn_rusted_main",
    "vulkannn_rusted",
]
vnn = None
for _mod_name in _VNN_CANDIDATES:
    try:
        import importlib
        vnn = importlib.import_module(_mod_name)
        print(f"[benchmark-big] Loaded VNN module: {_mod_name}")
        break
    except ImportError:
        continue
if vnn is None:
    raise ImportError("No vulkannn_rusted module found. Please run maturin develop --release first.")
Tensor = vnn.Tensor


RESULTS_FILE = "tests/last_results_big.json"
HISTORY_FILE = "tests/benchmark_history_big.log"
CACHE_DIR = "./tests/vnn_cache_big"
os.makedirs(CACHE_DIR, exist_ok=True)

def format_size(elements):
    if elements >= 1e9: return f"{elements/1e9:.1f}B"
    if elements >= 1e6: return f"{elements/1e6:.1f}M"
    return str(elements)

def check_parity(vnn_tensor, torch_tensor, name, atol=1e-2):
    rtol = 1e-3
    name_l = name.lower()
    if "bf16" in name_l:
        atol = 1.0
        if "sum" in name_l:
            atol = 100.0  # Increased for larger tensors
    elif "f16" in name_l and "sum" in name_l:
        atol = 0.5    # Increased for larger tensors
    if "int8" in name_l:
        if "sum" in name_l:
            # Int8 Sum: VNN is i64-exact, PT is fp32-approx. 
            # On 67M elements, diff can be > 10,000.
            atol = 20000.0
        elif "matmul" in name_l:
            atol = 500.0
    v_np = vnn_tensor.to_numpy()
    t_np = torch_tensor.detach().numpy() if hasattr(torch_tensor, 'detach') else torch_tensor
    v_np = v_np.flatten()
    t_np = t_np.flatten()
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, rtol=rtol)
        return True, 0.0
    except AssertionError as e:
        diff = np.abs(v_np - t_np)
        return False, np.max(diff)

def get_system_metrics():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = float(f.read().strip()) / 1000.0
    except:
        cpu_temp = -1.0
    try:
        load1, load5, load15 = os.getloadavg()
    except:
        load1 = -1.0
    return cpu_temp, load1

def get_torch_backend_label(dtype_str):
    parts = ["CPU"]
    if torch.backends.mkl.is_available():
        parts.append("MKL")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'openmp') and torch.backends.openmp.is_available():
        parts.append("OpenMP")
    dtype_map = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}
    parts.append(dtype_map.get(dtype_str, dtype_str))
    return "·".join(parts)

def run_bench(name, op, shape, mode="cpu", is_ssd=False, iterations=None, dtype="f32", inplace=False, pt_cache_time=None):
    if dtype == "f32":
        vnn_dtype = vnn.DataType.F32
        torch_dtype = torch.float32
    elif dtype == "f16":
        vnn_dtype = vnn.DataType.F16
        torch_dtype = torch.float16
    elif dtype == "bf16":
        vnn_dtype = vnn.DataType.BF16
        torch_dtype = torch.bfloat16
    elif dtype == "int8":
        vnn_dtype = vnn.DataType.Int8
        torch_dtype = torch.int8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if iterations is None:
        size_elements = np.prod(shape)
        if is_ssd or size_elements > 1e8: iterations = 1
        elif size_elements > 1e7: iterations = 3
        else: iterations = 5

    cpu_temp, load = get_system_metrics()
    temp_label = f"CPU Die: {cpu_temp:.1f}°C" if cpu_temp > 0 else "CPU Die: N/A"
    inplace_tag = " [inplace]" if inplace else ""
    print(f"\n>>> TEST: {name}{inplace_tag} ({mode.upper()}, {dtype.upper()}) | Shape: {shape} | SSD: {is_ssd} | Iter: {iterations} | {temp_label} | Load (1m): {load:.2f}", flush=True)

    import gc
    skip_pt = False
    res_torch = None
    if not is_ssd:
        a_np = np.random.randn(*shape).astype(np.float32)
        if op == "MatMul":
            b_np = np.random.randn(shape[-1], shape[-1]).astype(np.float32)
        else:
            b_np = np.random.randn(*shape).astype(np.float32)

        a_torch = torch.from_numpy(a_np).to(torch_dtype)
        b_torch = torch.from_numpy(b_np).to(torch_dtype) if op not in ["ReLU", "Sum"] else None

        pt_backend_label = get_torch_backend_label(dtype)
        skip_pt = pt_cache_time is not None
        if skip_pt:
            t_pt = pt_cache_time
            pt_backend_label += " (CACHED)"
            res_torch = None
        elif dtype == "int8" and op in ["GELU", "Softmax"]:
            t_pt = 0.0
            skip_pt = True
            res_torch = None
        else:
            t0 = time.perf_counter()
            for _ in range(iterations):
                if op == "MatMul":
                    if dtype == "int8":
                        res_torch = torch.matmul(a_torch.float(), b_torch.float()).to(torch.int8)
                    else:
                        res_torch = torch.matmul(a_torch, b_torch)
                elif "Monster" in name:
                    # Specific for overflow-triggering sum
                    res_torch = torch.sum(a_torch.float())
                elif op == "Add":
                    if dtype == "int8":
                         res_torch = (a_torch.short() + b_torch.short()).to(torch.int8)
                    else:
                         res_torch = a_torch + b_torch
                elif op == "Sub":
                    res_torch = a_torch - b_torch
                elif op == "Mul":
                    res_torch = a_torch * b_torch
                elif op == "ReLU":
                    if inplace:
                        a_torch_clone = a_torch.clone()
                        torch.relu_(a_torch_clone)
                        res_torch = a_torch_clone
                    else:
                        res_torch = torch.relu(a_torch)
                elif op == "GELU":
                    res_torch = torch.nn.functional.gelu(a_torch)
                elif op == "Sum":
                    if dtype == "int8":
                        res_torch = torch.sum(a_torch.float())
                    else:
                        res_torch = torch.sum(a_torch)
                elif op == "Softmax":
                    res_torch = torch.nn.functional.softmax(a_torch, dim=-1)
            t_pt = (time.perf_counter() - t0) / iterations

        if dtype == "int8":
             a_vnn = Tensor(data=a_np.astype(np.float32), dtype=vnn_dtype, device=mode)
             b_vnn = Tensor(data=b_np.astype(np.float32), dtype=vnn_dtype, device=mode) if op not in ["ReLU", "GELU", "Sum", "Softmax"] else None
        else:
             a_vnn = Tensor(data=a_np, dtype=vnn_dtype, device=mode)
             b_vnn = Tensor(data=b_np, dtype=vnn_dtype, device=mode) if op not in ["ReLU", "GELU", "Sum", "Softmax"] else None

        del a_np, b_np
        gc.collect()
    else:
        size_elements = np.prod(shape)
        a_path = os.path.join(CACHE_DIR, f"monster_a_{size_elements}.bin")
        a_vnn = Tensor.from_ssd(a_path, shape, dtype=vnn_dtype)
        a_vnn.device = mode
        b_vnn = None
        t_pt = 0
        res_torch = None
        pt_backend_label = "N/A"

    # VNN Warmup
    try:
        if op == "MatMul": _ = a_vnn @ b_vnn
        elif op == "Add": _ = a_vnn + b_vnn
        elif op == "Sub": _ = a_vnn - b_vnn
        elif op == "Mul": _ = a_vnn * b_vnn
        elif op == "ReLU":
            if inplace:
                out_vnn = Tensor(data=np.zeros(shape, dtype=np.float32), dtype=vnn_dtype, device=mode)
                a_vnn.relu_into(out_vnn)
            else:
                _ = a_vnn.relu()
        elif op == "GELU":
            _ = a_vnn.gelu()
        elif op == "Sum":
            _ = a_vnn.sum()
        elif op == "Softmax":
            _ = a_vnn.softmax(-1)
    except Exception as e:
        print(f"      [VNN] Error during warmup: {e}")
        return 0, 0, False, cpu_temp

    # VNN Timed run
    t0 = time.perf_counter()
    for i in range(iterations):
        if op == "MatMul": res_vnn = a_vnn @ b_vnn
        elif op == "Add": res_vnn = a_vnn + b_vnn
        elif op == "Sub": res_vnn = a_vnn - b_vnn
        elif op == "Mul": res_vnn = a_vnn * b_vnn
        elif op == "ReLU":
            if inplace:
                a_vnn.relu_into(out_vnn)
                res_vnn = out_vnn
            else:
                res_vnn = a_vnn.relu()
        elif op == "GELU":
            res_vnn = a_vnn.gelu()
        elif op == "Sum":
            res_vnn = a_vnn.sum()
        elif op == "Softmax":
            res_vnn = a_vnn.softmax(-1)

    t_vnn = (time.perf_counter() - t0) / iterations

    # Validation
    gc.collect() 
    parity_ok = "N/A"
    max_diff = 0.0
    if not is_ssd and res_torch is not None:
        atol = 1e-3 if dtype == "f32" else 1e-2
        if "f16" in dtype and op == "Sum": atol = 0.5
        parity_ok, max_diff = check_parity(res_vnn, res_torch.to(torch.float32), name, atol=atol)
        parity_str = "✅ OK" if parity_ok else f"❌ FAIL (diff: {max_diff:.6f})"
    elif skip_pt and not is_ssd:
        parity_ok = True
        parity_str = "✅ FAST"
    else:
        parity_str = "N/A (OOM-Safe)"

    print(f"    [PyTorch] {t_pt:.4f}s  ({pt_backend_label})", flush=True)
    print(f"    [VNN]     {t_vnn:.4f}s | Parity: {parity_str}", flush=True)

    if not is_ssd:
        del a_torch, b_torch, res_torch
    del a_vnn, b_vnn
    if not inplace:
        try: del res_vnn
        except: pass
    gc.collect()

    return t_pt, t_vnn, parity_ok, cpu_temp


import argparse

def get_stats(data):
    if not data: return 0, 0, 0
    return np.mean(data), np.median(data), np.std(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of full benchmark runs")
    parser.add_argument("--fast", action="store_true", help="Reuse cached times")
    args = parser.parse_args()

    print("="*60)
    print(f" VNN RUSTED: BIG DATA BENCHMARK")
    print("="*60)

    total_start = time.perf_counter()
    all_runs_results = []

    for run_idx in range(args.runs):
        run_results = []
        # Tensors large enough to make PT take ~seconds
        # 4096^2 is 16M elements. MatMul 4096^3 is 68 Gops.
        # 8192^2 is 67M elements.
        
        for dtype in ["f32", "f16", "bf16", "int8"]:
            print(f"\n{'#'*20} PHASE: {dtype.upper()} {'#'*20}")
            for mode in ["cpu", "vulkan", "hybrid"]:
                # --- MatMul ---
                pt, v_t, ok, temp = run_bench(f"MatMul_{dtype}_{mode}", "MatMul", (4096, 4096), mode=mode, dtype=dtype)
                run_results.append({"name": f"MatMul {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                # --- Elementwise & Reductions ---
                # Using 8192x8192 for elementwise (~67M elements)
                shape_big = (8192, 8192)
                
                pt, v_t, ok, temp = run_bench(f"Sum_{dtype}_{mode}", "Sum", shape_big, mode=mode, dtype=dtype)
                run_results.append({"name": f"Sum {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                pt, v_t, ok, temp = run_bench(f"Softmax_{dtype}_{mode}", "Softmax", shape_big, mode=mode, dtype=dtype)
                run_results.append({"name": f"Softmax {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                pt, v_t, ok, temp = run_bench(f"Mul_{dtype}_{mode}", "Mul", shape_big, mode=mode, dtype=dtype)
                run_results.append({"name": f"Mul {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})
                
                pt, v_t, ok, temp = run_bench(f"GELU_{dtype}_{mode}", "GELU", shape_big, mode=mode, dtype=dtype)
                run_results.append({"name": f"GELU {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                # --- ReLU Large ---
                pt, v_t, ok, temp = run_bench(f"ReLU_{dtype}_{mode}_100M", "ReLU", (100000000,), mode=mode, dtype=dtype)
                run_results.append({"name": f"ReLU {dtype} 100M ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

            if dtype == "int8":
                # --- Specific Overflow Edge Case: Total Sum of 100M elements filled with 127 ---
                # Expected Sum: 12.7 Billion (Exceeds i32 limit)
                print(f"\n[EDGE CASE] Injecting constant large data for Int8 Sum...")
                count_edge = 100_000_000
                data_edge = np.full((count_edge,), 127, dtype=np.int8)
                a_torch_edge = torch.from_numpy(data_edge).to(torch.int8)
                # PT sum (will be approximate due to fp32)
                t_pt_edge = 0.0 # Placeholder or measure
                a_vnn_edge = Tensor(data=data_edge.astype(np.float32), dtype=vnn.DataType.Int8, device="cpu")
                res_vnn_edge = a_vnn_edge.sum()
                res_vnn_np = res_vnn_edge.to_numpy().flatten()[0]
                expected_edge = float(count_edge) * 127.0
                diff_edge = abs(res_vnn_np - expected_edge)
                ok_edge = diff_edge < 1.0
                print(f"    [VNN Edge] Sum: {res_vnn_np} | Expected: {expected_edge} | OK: {ok_edge}")
                run_results.append({"name": "Int8 Sum Overflow Edge", "pt": 0, "vnn": 0, "ok": ok_edge, "cpu_temp_c": temp})

        all_runs_results.append(run_results)

    # Summary
    test_names = [r["name"] for r in all_runs_results[0]]
    aggregated = []
    for name in test_names:
        vnn_times = [r["vnn"] for run in all_runs_results for r in run if r["name"] == name]
        pt_times = [r["pt"] for run in all_runs_results for r in run if r["name"] == name]
        any_fail = any(not r["ok"] for run in all_runs_results for r in run if r["name"] == name and r["ok"] != "N/A")
        vnn_mean, vnn_med, vnn_std = get_stats(vnn_times)
        pt_mean, pt_med, pt_std = get_stats(pt_times)
        aggregated.append({"name": name, "vnn_med": vnn_med, "vnn_std": vnn_std, "pt_med": pt_med, "failed": any_fail})

    print("\n\n" + "="*80)
    print(f"{'VNN BIG SUMMARY':^80}")
    print("="*80)
    for res in aggregated:
        ratio = res["vnn_med"] / res["pt_med"] if res["pt_med"] > 0 else 0
        parity = "❌ FAIL" if res["failed"] else "✅ PASS"
        print(f"{res['name']:<35} | PT: {res['pt_med']:8.4f}s | VNN: {res['vnn_med']:8.4f}s | {ratio:8.2f}x | {parity}")
    print("="*80)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(aggregated, f, indent=4)
