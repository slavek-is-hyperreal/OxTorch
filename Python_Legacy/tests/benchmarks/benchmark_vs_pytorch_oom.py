import Python_Legacy.vulkan_nn_lib.torch_shim as vnn
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

def benchmark_oom():
    # Target: 30GB Tensor (Exceeds ~19GB usable RAM)
    # N = 30 * 1024^3 / 4 bytes (for float32)
    N = 8_000_000_000 # ~32GB
    
    print(f"\n--- Comparative Benchmark: VNN vs PyTorch (32GB Tensor) ---")
    print(f"System Info: Available RAM approx {get_available_ram()/1e9:.1f}GB (User noted 5GB reserved for ZFS ARC)")
    
    # 1. VNN SSD-Native
    print(f"\n[1/2] VNN (SSD-Native) initialized...")
    try:
        t0 = time.perf_counter()
        # VNN zeros() on SSD is fast (deferred)
        v_a = vnn.zeros(N, device='ssd')
        # Fill some values to make sum non-zero
        v_a.arr[0:1000] = 1.0
        
        print("      Calculating Fused Sum...")
        t_sum = time.perf_counter()
        v_res = v_a.fused_sum(1.5, op='mul')
        res_val = v_res.item()
        t_end = time.perf_counter()
        
        print(f"      VNN Result: {res_val:.1f}")
        print(f"      VNN Total Time: {t_end - t0:.2f}s (Sum: {t_end-t_sum:.2f}s)")
    except Exception as e:
        print(f"      VNN FAILED: {e}")

    # 2. PyTorch CPU
    print(f"\n[2/2] PyTorch (CPU) initialized...")
    try:
        t0 = time.perf_counter()
        # This will likely OOM or cause heavy thrashing
        p_a = torch.zeros(N, dtype=torch.float32)
        p_a[0:1000] = 1.0
        
        print("      Calculating Sum...")
        t_sum = time.perf_counter()
        p_res = (p_a * 1.5).sum()
        res_val = p_res.item()
        t_end = time.perf_counter()
        
        print(f"      PyTorch Result: {res_val:.1f}")
        print(f"      PyTorch Total Time: {t_end - t0:.2f}s")
    except RuntimeError as e:
        print(f"      PyTorch FAILED: {e}")
    except Exception as e:
        print(f"      PyTorch Error (likely OOM/Killed): {e}")

if __name__ == "__main__":
    benchmark_oom()
