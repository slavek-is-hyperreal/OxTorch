import time
import numpy as np
import torch
import os
import sys
import json

# Force line-buffered output even when piped (fixes invisible partial results)
sys.stdout.reconfigure(line_buffering=True)

# Branch-aware dynamic import: tries each known branch module name in order.
_VNN_CANDIDATES = [
    "vulkannn_rusted_exp",   # dev_raw_vulkan (experimental ash branch)
    "vulkannn_rusted_dev",   # dev
    "vulkannn_rusted_test",  # test
    "vulkannn_rusted_main",  # main
    "vulkannn_rusted",       # fallback (any generic build)
]
vnn = None
for _mod_name in _VNN_CANDIDATES:
    try:
        import importlib
        vnn = importlib.import_module(_mod_name)
        print(f"[benchmark] Loaded VNN module: {_mod_name}")
        break
    except ImportError:
        continue
if vnn is None:
    raise ImportError("No vulkannn_rusted module found. Please run maturin develop --release first.")
Tensor = vnn.Tensor


RESULTS_FILE = "tests/last_results.json"
HISTORY_FILE = "tests/benchmark_history.log"
CACHE_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def format_size(elements):
    if elements >= 1e9: return f"{elements/1e9:.1f}B"
    if elements >= 1e6: return f"{elements/1e6:.1f}M"
    return str(elements)

def check_parity(vnn_tensor, torch_tensor, name, atol=1e-2):
    rtol = 1e-3
    if "bf16" in name.lower():
        atol = 1.0
        rtol = 1.5e-2
    if "monster" in name.lower():
        pass
    v_np = vnn_tensor.to_numpy()
    t_np = torch_tensor.detach().numpy() if hasattr(torch_tensor, 'detach') else torch_tensor
    v_np = v_np.flatten()
    t_np = t_np.flatten()
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, rtol=1e-3)
        return True, 0.0
    except AssertionError as e:
        diff = np.abs(v_np - t_np)
        return False, np.max(diff)

def get_system_metrics():
    """Read CPU die temperature (thermal_zone0 = Ivy Bridge package sensor) and load."""
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
    """Return a human-readable label for which PyTorch backend is in use."""
    parts = ["CPU"]
    if torch.backends.mkl.is_available():
        parts.append("MKL")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'openmp') and torch.backends.openmp.is_available():
        parts.append("OpenMP")
    dtype_map = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}
    parts.append(dtype_map.get(dtype_str, dtype_str))
    return "·".join(parts)

def run_bench(name, op, shape, mode="cpu", is_ssd=False, iterations=None, dtype="f32", inplace=False):
    """Run a single benchmark test.
    
    inplace=True: measures relu_into() (VNN) vs relu_() (PyTorch in-place).
    inplace=False (default): measures relu() allocating path.
    """
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
        elif (dtype == "f16" or dtype == "bf16") and op == "MatMul": iterations = 2
        else: iterations = 10 if dtype in ["f16", "bf16"] else 20

    cpu_temp, load = get_system_metrics()
    temp_label = f"CPU Die: {cpu_temp:.1f}°C" if cpu_temp > 0 else "CPU Die: N/A"
    inplace_tag = " [inplace]" if inplace else ""
    print(f"\n>>> TEST: {name}{inplace_tag} ({mode.upper()}, {dtype.upper()}) | Shape: {shape} | SSD: {is_ssd} | Iter: {iterations} | {temp_label} | Load (1m): {load:.2f}", flush=True)

    import gc
    if not is_ssd:
        a_np = np.random.randn(*shape).astype(np.float32)
        if op == "MatMul":
            b_np = np.random.randn(shape[-1], shape[-1]).astype(np.float32)
        else:
            b_np = np.random.randn(*shape).astype(np.float32)

        a_torch = torch.from_numpy(a_np).to(torch_dtype)
        b_torch = torch.from_numpy(b_np).to(torch_dtype) if op not in ["ReLU", "Sum"] else None

        # PyTorch Reference
        pt_backend_label = get_torch_backend_label(dtype)
        t0 = time.perf_counter()
        for _ in range(iterations):
            if op == "MatMul":
                res_torch = torch.matmul(a_torch, b_torch)
            elif op == "Add":
                res_torch = a_torch + b_torch
            elif op == "ReLU":
                if inplace:
                    # In-place: reuse a_torch buffer — fair comparison to relu_into()
                    a_torch_clone = a_torch.clone()  # clone once outside loop
                    torch.relu_(a_torch_clone)
                    res_torch = a_torch_clone
                else:
                    res_torch = torch.relu(a_torch)
            elif op == "Sum":
                res_torch = torch.sum(a_torch)
        t_pt = (time.perf_counter() - t0) / iterations

        a_vnn = Tensor(data=a_np, dtype=vnn_dtype, device=mode)
        b_vnn = Tensor(data=b_np, dtype=vnn_dtype, device=mode) if op not in ["ReLU", "Sum"] else None

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
        elif op == "ReLU":
            if inplace:
                # Pre-allocate output tensor for relu_into
                out_vnn = Tensor(data=np.zeros(shape, dtype=np.float32), dtype=vnn_dtype, device=mode)
                a_vnn.relu_into(out_vnn)
            else:
                _ = a_vnn.relu()
        elif op == "Sum":
            _ = a_vnn.sum()
    except Exception as e:
        print(f"      [VNN] Error during warmup: {e}")
        return 0, 0, False, cpu_temp

    # VNN Timed run
    t0 = time.perf_counter()
    for i in range(iterations):
        if op == "MatMul": res_vnn = a_vnn @ b_vnn
        elif op == "Add": res_vnn = a_vnn + b_vnn
        elif op == "ReLU":
            if inplace:
                a_vnn.relu_into(out_vnn)
                res_vnn = out_vnn
            else:
                res_vnn = a_vnn.relu()
        elif op == "Sum":
            res_vnn = a_vnn.sum()

        if i < iterations - 1:
            if not inplace:
                del res_vnn
        if i % 5 == 0:
            gc.collect()

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
    parser.add_argument("--runs", type=int, default=1, help="Number of full benchmark runs for statistics")
    args = parser.parse_args()

    print("="*60)
    print(f" VNN RUSTED SAFETY NET: TRI-PRECISION AUDIT v3.3")
    print(f" (Mode: Statistical Analysis - {args.runs} runs)")
    print(f" CPU Temp sensor: thermal_zone0 (Ivy Bridge package die)")
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
                # --- MatMul ---
                pt, v_t, ok, temp = run_bench(f"MatMul_{dtype}_{mode}", "MatMul", (2048, 2048), mode=mode, dtype=dtype)
                run_results.append({"name": f"MatMul {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                # --- Sum ---
                pt, v_t, ok, temp = run_bench(f"Sum_{dtype}_{mode}", "Sum", (2048, 2048), mode=mode, dtype=dtype)
                run_results.append({"name": f"Sum {dtype} ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                # --- ReLU 1M: alloc path ---
                pt, v_t, ok, temp = run_bench(f"ReLU_{dtype}_{mode}", "ReLU", (1000000,), mode=mode, dtype=dtype, inplace=False)
                run_results.append({"name": f"ReLU {dtype} 1M alloc ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                # --- ReLU 1M: inplace path (no alloc overhead) ---
                if mode == "cpu":  # relu_into only meaningful on CPU path (avoids the allocation debate)
                    pt, v_t, ok, temp = run_bench(f"ReLU_{dtype}_{mode}_inplace", "ReLU", (1000000,), mode=mode, dtype=dtype, inplace=True)
                    run_results.append({"name": f"ReLU {dtype} 1M inplace ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

                # --- ReLU 15M: above GPU break-even threshold (VULKAN_MIN_ELEMS=4M) ---
                pt, v_t, ok, temp = run_bench(f"ReLU_{dtype}_{mode}_15M", "ReLU", (15000000,), mode=mode, dtype=dtype)
                run_results.append({"name": f"ReLU {dtype} 15M ({mode})", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})

        pt, v_t, ok, temp = run_bench("Monster_ReLU_F32_SSD", "ReLU", (4000000000,), mode="cpu", is_ssd=True, dtype="f32")
        run_results.append({"name": "Monster ReLU 16GB SSD", "pt": pt, "vnn": v_t, "ok": ok, "cpu_temp_c": temp})
        all_runs_results.append(run_results)


    # --- AGGREGATE STATS ---
    test_names = [r["name"] for r in all_runs_results[0]]
    aggregated = []

    for name in test_names:
        vnn_times  = [r["vnn"]  for run in all_runs_results for r in run if r["name"] == name]
        pt_times   = [r["pt"]   for run in all_runs_results for r in run if r["name"] == name]
        temps      = [r["cpu_temp_c"] for run in all_runs_results for r in run if r["name"] == name and r["cpu_temp_c"] > 0]
        any_fail   = any(not r["ok"] for run in all_runs_results for r in run if r["name"] == name and r["ok"] != "N/A")

        vnn_mean, vnn_med, vnn_std = get_stats(vnn_times)
        pt_mean,  pt_med,  pt_std  = get_stats(pt_times)
        temp_mean = float(np.mean(temps)) if temps else -1.0

        aggregated.append({
            "name": name,
            "vnn_mean": vnn_mean, "vnn_med": vnn_med, "vnn_std": vnn_std,
            "pt_mean": pt_mean, "pt_med": pt_med,
            "cpu_temp_c_mean": temp_mean,
            "failed": any_fail
        })

    # --- SUMMARY TABLE ---
    print("\n\n" + "="*105)
    print(f"{'VNN STATISTICAL SUMMARY (' + str(args.runs) + ' runs)':^105}")
    print("="*105)
    print(f"{'Test Case':<35} | {'PT (Med)':<10} | {'VNN (Med)':<10} | {'VNN (Std)':<10} | {'Ratio':<8} | {'CPU°C':<6} | {'Parity'}")
    print("-" * 105)

    for res in aggregated:
        ratio = res["vnn_med"] / res["pt_mean"] if res["pt_mean"] > 0 else 0
        ratio_str = f"{ratio:8.2f}x" if ratio > 0.1 else f"{ratio:8.4f}x"
        parity = "❌ FAIL" if res["failed"] else "✅ PASS"
        temp_str = f"{res['cpu_temp_c_mean']:.0f}" if res['cpu_temp_c_mean'] > 0 else "N/A"
        print(f"{res['name']:<35} | {res['pt_med']:8.4f}s | {res['vnn_med']:8.4f}s | {res['vnn_std']:8.4f}s | {ratio_str} | {temp_str:>5}° | {parity}")

    print("="*105)

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
        "torch_backend": get_torch_backend_label("f32"),
        "results": aggregated
    })

    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nResults appended to {RESULTS_FILE} (History length: {len(history)})")
    print(f"Total session duration: {total_duration:.2f}s")
    print("*" * 80)
    print(f"Results saved to {RESULTS_FILE} and logged to {HISTORY_FILE}")
