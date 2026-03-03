import time
import numpy as np
import torch
import os
from vulkannn_rusted import Tensor

def format_size(elements):
    if elements >= 1e9: return f"{elements/1e9:.1f}B"
    if elements >= 1e6: return f"{elements/1e6:.1f}M"
    return str(elements)

def run_matmul_bench(m, k, n, is_ssd=False):
    print(f"\n--- Benchmarking MatMul | {m}x{k}x{n} | SSD: {is_ssd} ---")
    
    if not is_ssd:
        a_np = np.random.randn(m, k).astype(np.float32)
        b_np = np.random.randn(k, n).astype(np.float32)
        
        # PyTorch
        t0 = time.time()
        c_pt = torch.matmul(torch.from_numpy(a_np), torch.from_numpy(b_np))
        t_pt = time.time() - t0
        print(f"  [PyTorch]  Execution: {t_pt:.4f}s")
        
        # VNN RAM
        a_vnn = Tensor(a_np, device="cpu")
        b_vnn = Tensor(b_np, device="cpu")
    else:
        # SSD Path
        a_path = "/vectorlegis_ssd_pool/vnn_cache/unified_matmul_a.bin"
        b_path = "/vectorlegis_ssd_pool/vnn_cache/unified_matmul_b.bin"
        
        if not os.path.exists(a_path) or os.path.getsize(a_path) != m*k*4:
            print(f"  Generating SSD data...")
            a_np = np.random.randn(m, k).astype(np.float32)
            a_np.tofile(a_path)
            b_np = np.random.randn(k, n).astype(np.float32)
            b_np.tofile(b_path)
            
        a_vnn = Tensor.from_ssd(a_path, [m, k])
        b_vnn = Tensor.from_ssd(b_path, [k, n])
        a_vnn.device = "cpu"
        b_vnn.device = "cpu"
        
        # We don't bench PyTorch on SSD directly (it swaps anyway)
        t_pt = 1.0 # placeholder or previous run

    t0 = time.time()
    c_vnn = a_vnn @ b_vnn
    t_vnn = time.time() - t0
    print(f"  [VNN CPU]  Execution: {t_vnn:.4f}s")
    
    if not is_ssd:
        parity = np.allclose(c_vnn.to_numpy(), c_pt.numpy(), atol=1e-2)
        print(f"  {'✅ PARITY OK' if parity else '❌ PARITY FAIL'}")
        return t_pt, t_vnn
    return 0, t_vnn

def run_ops_bench(size, op, mode):
    print(f"\n--- Benchmarking {op} | {format_size(size)} | Mode: {mode} ---")
    
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)
    
    # PyTorch
    at = torch.from_numpy(a_np)
    bt = torch.from_numpy(b_np)
    t0 = time.time()
    if op == "Add": res_pt = at + bt
    elif op == "ReLU": res_pt = torch.relu(at)
    t_pt = time.time() - t0
    print(f"  [PyTorch]  Execution: {t_pt:.4f}s")
    
    # VNN
    a_vnn = Tensor(a_np, device=mode)
    b_vnn = Tensor(b_np, device=mode)
    
    # Check if we need SSD streaming for VNN
    is_oom = (size * 4 * 3) > 12e9 # Roughly > 12GB
    if is_oom:
        print("  [OOM Mode] Streaming to SSD...")
        out_path = "/vectorlegis_ssd_pool/vnn_cache/bench_out.bin"
        out_vnn = Tensor.new_ssd(out_path, [size])
    else:
        out_vnn = Tensor(shape=[size], device=mode)
        
    t0 = time.time()
    if op == "Add": a_vnn.add_into(b_vnn, out_vnn)
    elif op == "ReLU": a_vnn.relu_into(out_vnn)
    t_vnn = time.time() - t0
    print(f"  [VNN {mode}] Execution: {t_vnn:.4f}s")
    
    # Parity
    if not is_oom:
        parity = np.allclose(out_vnn.to_numpy(), res_pt.numpy(), atol=1e-3)
        print(f"  {'✅ PARITY OK' if parity else '❌ PARITY FAIL'}")
        
    return t_pt, t_vnn

if __name__ == "__main__":
    os.makedirs("/vectorlegis_ssd_pool/vnn_cache", exist_ok=True)
    
    results = []
    
    # 1. MatMul RAM (10k)
    pt, vnn = run_matmul_bench(10000, 10000, 10000)
    results.append(("MatMul 10k CPU", pt, vnn))
    
    # 2. Add CPU (250M)
    pt, vnn = run_ops_bench(250000000, "Add", "cpu")
    results.append(("Add 250M CPU", pt, vnn))
    
    # 3. ReLU CPU (250M)
    pt, vnn = run_ops_bench(250000000, "ReLU", "cpu")
    results.append(("ReLU 250M CPU", pt, vnn))
    
    # 4. Add Vulkan (250M)
    pt, vnn = run_ops_bench(250000000, "Add", "vulkan")
    results.append(("Add 250M Vulkan", pt, vnn))
    
    # 5. Massive OOM Test (Optional but good)
    # _, vnn = run_matmul_bench(40000, 40000, 40000, is_ssd=True)
    # results.append(("MatMul 40k SSD", 2000, vnn)) # Approx PyTorch placeholder

    print("\n\n=== UNIFIED BENCHMARK SUMMARY ===")
    print(f"{'Test Case':<25} | {'PyTorch':<10} | {'VNN':<10} | {'Ratio':<10}")
    print("-" * 65)
    for name, pt, vnn in results:
        ratio = vnn / pt if pt > 0 else 0
        print(f"{name:<25} | {pt:>8.4f}s | {vnn:>8.4f}s | {ratio:>8.2f}x")
