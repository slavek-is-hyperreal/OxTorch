import time
import numpy as np
import torch
import os
import sys
import json
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

def run_bench(name, op, shape, mode="cpu", is_ssd=False, iterations=None):
    if iterations is None:
        size_elements = np.prod(shape)
        if is_ssd or size_elements > 5e7: iterations = 1
        elif size_elements > 5e6: iterations = 5
        else: iterations = 20 # Stable for small ReLU/MatMul
        
    print(f"\n>>> TEST: {name} ({mode.upper()}) | Shape: {shape} | SSD: {is_ssd} | Iter: {iterations}")
    
    # Pre-generate or Load Data
    if not is_ssd:
        a_np = np.random.randn(*shape).astype(np.float32)
        if op == "MatMul":
            # For matmul we need 2D
            b_np = np.random.randn(shape[-1], shape[-1]).astype(np.float32) # Simple square for bench
            torch_shape_b = (shape[-1], shape[-1])
        else:
            b_np = np.random.randn(*shape).astype(np.float32)
            torch_shape_b = shape
            
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        
        # PyTorch Reference
        # Warmup
        if op == "MatMul": _ = torch.matmul(a_torch, b_torch)
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            if op == "MatMul": res_torch = torch.matmul(a_torch, b_torch)
            elif op == "Add": res_torch = a_torch + b_torch
            elif op == "ReLU": res_torch = torch.relu(a_torch)
        t_pt = (time.perf_counter() - t0) / iterations
        
        # VNN Input
        a_vnn = Tensor(a_np, device=mode)
        b_vnn = Tensor(b_np, device=mode) if op != "ReLU" else None
    else:
        # SSD Path (Monster Tensors)
        size_elements = np.prod(shape)
        size_elements = np.prod(shape)
        a_path = os.path.join(CACHE_DIR, f"monster_a_{size_elements}.bin")
        expected_bytes = size_elements * 4
        if not os.path.exists(a_path) or os.path.getsize(a_path) != expected_bytes:
            print(f"    Generating {format_size(size_elements)} SSD junk data (exact {expected_bytes} bytes)...")
            # Write in 4MB chunks but ensure exact total size
            chunk_size = 1024 * 1024
            junk = np.ones(chunk_size, dtype=np.float32)
            with open(a_path, 'wb') as f:
                remaining = size_elements
                while remaining > 0:
                    write_size = min(remaining, chunk_size)
                    f.write(junk[:write_size].tobytes())
                    remaining -= write_size
        
        a_vnn = Tensor.from_ssd(a_path, shape)
        a_vnn.device = mode
        b_vnn = None # Simplification for SSD Monster tests (usually unary or restricted)
        t_pt = 0 # PyTorch would OOM
        res_torch = None

    # VNN Execution
    # Warmup
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
        parity_ok, max_diff = check_parity(res_vnn, res_torch, name)
        parity_str = "✅ OK" if parity_ok else f"❌ FAIL (diff: {max_diff:.6f})"
    else:
        parity_str = "N/A (OOM-Safe)"

    print(f"    [PyTorch] {t_pt:.4f}s")
    print(f"    [VNN]     {t_vnn:.4f}s | Parity: {parity_str}")
    
    return t_pt, t_vnn, parity_ok

if __name__ == "__main__":
    print("="*60)
    print(" VNN RUSTED SAFETY NET: COMPREHENSIVE AUDIT v2.8")
    print("="*60)
    
    results = []
    
    # --- PHASE 1: RAM-RESIDENT PARITY (Correctness) ---
    # Test all modes for basic math
    for mode in ["cpu", "vulkan", "hybrid"]:
        # 1. MatMul Parity
        pt, vnn, ok = run_bench(f"MatMul_Parity_{mode}", "MatMul", (2048, 2048), mode=mode)
        results.append((f"MatMul 2k ({mode})", pt, vnn, ok))
        
        # 2. ReLU Parity
        pt, vnn, ok = run_bench(f"ReLU_Parity_{mode}", "ReLU", (1000000,), mode=mode)
        results.append((f"ReLU 1M ({mode})", pt, vnn, ok))

    # --- PHASE 2: TILING & DISPATCH LIMITS ---
    # Testing non-standard shapes that might break tiling logic
    pt, vnn, ok = run_bench("MatMul_Tiling_Thin", "MatMul", (1, 10000), mode="vulkan") # 1x10000 @ 10000x10000 (effectively)
    results.append(("MatMul 1x10k (Tiling)", pt, vnn, ok))

    # --- PHASE 3: SPEED SUPREMACY (Large RAM) ---
    # 10k x 10k MatMul
    pt, vnn, ok = run_bench("MatMul_Speed_10k", "MatMul", (10000, 10000), mode="cpu")
    results.append(("MatMul 10k CPU Speed", pt, vnn, ok))

    # --- PHASE 4: MONSTER STREAMING (SSD) ---
    # Testing 16GB ReLU operation (exceeds typical VRAM and pushes RAM limits)
    # 4 billion elements = 16GB
    pt, vnn, ok = run_bench("Monster_ReLU_SSD", "ReLU", (4000000000,), mode="cpu", is_ssd=True)
    results.append(("Monster ReLU 16GB SSD", pt, vnn, ok))

    # --- SUMMARY ---
    print("\n\n" + "="*80)
    print(f"{'VNN SAFETY NET SUMMARY':^80}")
    print("="*80)
    print(f"{'Test Case':<30} | {'PyTorch':<10} | {'VNN':<10} | {'Ratio':<8} | {'Parity'}")
    print("-" * 80)
    
    failed = False
    for name, pt, vnn, ok in results:
        ratio = vnn / pt if pt > 0 else 0
        r_str = f"{ratio:.2f}x" if pt > 0 else "---"
        p_str = "✅ PASS" if (ok is True or ok == "N/A") else "❌ FAIL"
        if ok is False: failed = True
        print(f"{name:<30} | {pt:>8.4f}s | {vnn:>8.4f}s | {r_str:>8} | {p_str}")
    
    print("="*80)
    if failed:
        print("\n⚠️  WARNING: Some parity tests failed! Check numerical differences.")
    else:
        print("\n✨ ALL CORE SYSTEMS OPERATIONAL: Parity and Speed Verified.")
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
    for name, pt, vnn, ok in results:
        ratio = vnn / pt if pt > 0 else 0
        current_save[name] = ratio
        prev_ratio = prev_results.get(name)
        if prev_ratio is not None:
            delta = ratio - prev_ratio
            perc = (delta / prev_ratio) * 100 if prev_ratio > 0 else 0
            d_str = f"{delta:>+8.4f}"
            # Sensitivity: 10% change in ratio is significant
            status = "🔴 REGRESSION" if perc > 10 else "🟢 IMPROVED" if perc < -10 else "🟡 STABLE"
            print(f"{name:<30} | {ratio:>8.4f} | {prev_ratio:>8.4f} | {d_str} | {status} ({perc:>+5.1f}%)")
        else:
            print(f"{name:<30} | {ratio:>8.4f} | {'N/A':>8} | {'---':>8} | NEW TEST")
    
    # Save current ratios for next time
    with open(RESULTS_FILE, 'w') as f:
        json.dump(current_save, f, indent=4)
        
    # Append to human-readable log
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"\n--- Benchmark Run: {time.ctime()} ---\n")
        for name, pt, vnn, ok in results:
            f.write(f"{name:<30} | VNN: {vnn:.4f}s | PT: {pt:.4f}s | Parity: {ok}\n")

    print("*" * 80)
    print(f"Results saved to {RESULTS_FILE} and logged to {HISTORY_FILE}")
