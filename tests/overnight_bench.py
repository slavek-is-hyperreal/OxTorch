import time
import numpy as np
import torch
import os
import sys
from vulkannn_rusted import Tensor

# Settings
ITERATIONS_SMALL = 5000
ITERATIONS_HEAVY = 100 # For MatMul 10k
ITERATIONS_MONSTER = 5  # For 16GB SSD test
CACHE_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def check_parity_silent(vnn_tensor, torch_tensor, atol=1e-3):
    v_np = vnn_tensor.to_numpy().flatten()
    t_np = torch_tensor.detach().numpy().flatten() if hasattr(torch_tensor, 'detach') else torch_tensor.flatten()
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, rtol=1e-3)
        return True, 0.0
    except AssertionError:
        diff = np.abs(v_np - t_np)
        return False, np.max(diff)

def run_stress_test(name, op, shape, mode="cpu", is_ssd=False, iterations=100):
    print(f"\n[STRESS] {name} ({mode.upper()}) | Iterations: {iterations} | Shape: {shape}")
    
    # 1. Prepare Data
    if not is_ssd:
        a_np = np.random.randn(*shape).astype(np.float32)
        if op == "MatMul":
            b_np = np.random.randn(shape[-1], shape[-1]).astype(np.float32)
            b_torch = torch.from_numpy(b_np)
        else:
            b_np = None
            b_torch = None
        a_torch = torch.from_numpy(a_np)
    else:
        # SSD Monster setup (simplified, just ones)
        size_elements = np.prod(shape)
        a_path = os.path.join(CACHE_DIR, f"monster_a_{size_elements}.bin")
        if not os.path.exists(a_path):
            with open(a_path, 'wb') as f:
                f.write(np.ones(1024*1024, dtype=np.float32).tobytes() * (size_elements // (1024*1024)))
        a_vnn = Tensor.from_ssd(a_path, shape)
        a_vnn.device = mode
        return 0.0, 0.0, True # Skip heavy PT bench for monster

    # VNN & PT Setup
    a_vnn = Tensor(a_np, device=mode)
    b_vnn = Tensor(b_np, device=mode) if op == "MatMul" else None

    times_pt = []
    times_vnn = []
    parity_errors = 0

    # 2. Loop
    for i in range(iterations):
        # PyTorch
        t0 = time.perf_counter()
        if op == "MatMul": _ = torch.matmul(a_torch, b_torch)
        elif op == "ReLU": _ = torch.relu(a_torch)
        times_pt.append(time.perf_counter() - t0)

        # VNN
        t0 = time.perf_counter()
        if op == "MatMul": res_vnn = a_vnn @ b_vnn
        elif op == "ReLU": res_vnn = a_vnn.relu()
        times_vnn.append(time.perf_counter() - t0)

        # Periodic Parity Check (every 500 or final)
        if i % 500 == 0 or i == iterations - 1:
            if op == "MatMul": res_torch = torch.matmul(a_torch, b_torch)
            else: res_torch = torch.relu(a_torch)
            ok, _ = check_parity_silent(res_vnn, res_torch)
            if not ok: parity_errors += 1

        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{iterations} | Avg VNN: {np.mean(times_vnn):.4f}s")

    # Statistics
    stats = {
        "pt_mean": np.mean(times_pt),
        "vnn_mean": np.mean(times_vnn),
        "vnn_std": np.std(times_vnn),
        "vnn_cv": (np.std(times_vnn) / np.mean(times_vnn)) * 100 if np.mean(times_vnn) > 0 else 0,
        "vnn_p95": np.percentile(times_vnn, 95),
        "parity_ok": parity_errors == 0
    }
    return stats

def main():
    print("="*80)
    print(f" VNN OVERNIGHT STRESS TEST v2.9 - {ITERATIONS_SMALL} Iterations Target")
    print("="*80)
    
    results = []
    
    # 1. Small/Medium Benchmarks (5000 runs)
    bench_configs = [
        ("MatMul 2k CPU", "MatMul", (2048, 2048), "cpu"),
        ("MatMul 2k Vulkan", "MatMul", (2048, 2048), "vulkan"),
        ("MatMul 2k Hybrid", "MatMul", (2048, 2048), "hybrid"),
        ("ReLU 1M CPU", "ReLU", (1000000,), "cpu"),
        ("ReLU 1M Vulkan", "ReLU", (1000000,), "vulkan"),
        ("MatMul 1x10k Tiling", "MatMul", (1, 10000), "vulkan"),
    ]

    for name, op, shape, mode in bench_configs:
        stats = run_stress_test(name, op, shape, mode, iterations=ITERATIONS_SMALL)
        results.append((name, stats))

    # 2. Heavy Benchmark (100 runs)
    stats = run_stress_test("MatMul 10k CPU Speed", "MatMul", (10000, 10000), "cpu", iterations=ITERATIONS_HEAVY)
    results.append(("MatMul 10k CPU Speed", stats))

    # Summary
    print("\n\n" + "="*80)
    print(f"{'VNN STRESS TEST FINAL SUMMARY':^80}")
    print("="*80)
    print(f"{'Test Case':<25} | {'Avg PT':<10} | {'Avg VNN':<10} | {'Ratio':<6} | {'CV %':<8} | {'P95':<8} | {'Parity'}")
    print("-" * 105)
    
    for name, s in results:
        pt = s["pt_mean"]
        vnn = s["vnn_mean"]
        ratio = vnn / pt if pt > 0 else 0
        r_str = f"{ratio:.2f}x" if pt > 0 else "---"
        p_str = "✅ PASS" if s["parity_ok"] else "❌ FAIL"
        
        # Stability check
        stability = s["vnn_cv"]
        st_str = f"{stability:>6.1f}%"
        p95_str = f"{s['vnn_p95']:>8.5f}s"
        
        print(f"{name:<25} | {pt:>8.4f}s | {vnn:>8.4f}s | {r_str:>6} | {st_str} | {p95_str} | {p_str}")
    
    print("="*80)
    print("✨ STRESS TEST COMPLETE. REST WELL.")
    print("="*80)

if __name__ == "__main__":
    main()
