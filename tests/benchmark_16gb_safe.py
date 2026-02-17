import vulkan_nn_lib.torch_shim as vnn
import torch
import numpy as np
import time
import os

def get_available_ram():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except:
        return 0

def benchmark_safe():
    # Target: 8GB Tensor (Should safely fit in ~16.9GB RAM)
    # N = 8 * 1024^3 / 4 bytes (for float32)
    N = 2_000_000_000 # ~8GB
    
    print(f"\n--- Comparative Benchmark: VNN vs PyTorch (8GB Tensor - 'Safe-RAM' Case) ---")
    print(f"System Info: Available RAM approx {get_available_ram()/1e9:.1f}GB")
    
    # 1. PyTorch CPU (The Speed Baseline)
    print(f"\n[1/3] PyTorch (CPU) initialization...")
    try:
        t0 = time.perf_counter()
        p_a = torch.zeros(N, dtype=torch.float32)
        p_a[0:1000] = 1.0
        t_init = time.perf_counter() - t0
        
        print("      Calculating (A * 1.5).sum()...")
        t_sum = time.perf_counter()
        p_res = (p_a * 1.5).sum()
        res_val = p_res.item()
        t_total = time.perf_counter() - t_sum
        
        print(f"      PyTorch Result: {res_val:.1f}")
        print(f"      PyTorch Time: {t_total:.2f}s (Init: {t_init:.2f}s)")
        p_time = t_total
    except Exception as e:
        print(f"      PyTorch FAILED: {e}")
        p_time = None
    
    # Explicitly release PyTorch memory for fair 'auto' comparison
    import gc
    try: del p_a
    except: pass
    gc.collect()

    # 2. VNN Auto (Likely picking RAM/NumPy now)
    print(f"\n[2/3] VNN (Auto Logic) initialization...")
    try:
        t0 = time.perf_counter()
        v_a = vnn.zeros(N, device='auto')
        v_a.arr[0:1000] = 1.0
        t_init = time.perf_counter() - t0
        
        print("      Calculating Fused Sum...")
        t_sum = time.perf_counter()
        v_res = v_a.fused_sum(1.5, op='mul')
        res_val = v_res.item()
        t_total = time.perf_counter() - t_sum
        
        print(f"      VNN (Auto) Result: {res_val:.1f}")
        print(f"      VNN (Auto) Time: {t_total:.2f}s (Init: {t_init:.2f}s) | Device: {v_a.device}")
        
        if p_time:
            print(f"      Ratio: x{t_total/p_time:.2f} vs PyTorch")
    except Exception as e:
        print(f"      VNN (Auto) FAILED: {e}")

    # 3. VNN Forced SSD (Stability mode)
    print(f"\n[3/3] VNN (Forced SSD) initialization...")
    try:
        t0 = time.perf_counter()
        v_s = vnn.zeros(N, device='ssd')
        v_s.arr[0:1000] = 1.0
        t_init = time.perf_counter() - t0
        
        print("      Calculating Fused Sum (SSD Streaming)...")
        t_sum = time.perf_counter()
        v_res_s = v_s.fused_sum(1.5, op='mul')
        res_val = v_res_s.item()
        t_total = time.perf_counter() - t_sum
        
        print(f"      VNN (SSD) Result: {res_val:.1f}")
        print(f"      VNN (SSD) Time: {t_total:.2f}s (Init: {t_init:.2f}s)")
        
        if p_time:
            print(f"      Ratio: x{t_total/p_time:.2f} vs PyTorch (The 'Security Tax' for OOM-Safety)")
    except Exception as e:
        print(f"      VNN (SSD) FAILED: {e}")

if __name__ == "__main__":
    benchmark_safe()
