import os
import numpy as np
import time
from vulkan_nn_lib import torch_shim as torch
from vulkan_nn_lib.tensor import Tensor
from vulkan_nn_lib.memory import MemoryManager

def benchmark_dominance():
    print("=== VNN Dominance Benchmark (PyTorch Killer) ===")
    
    # We want to demonstrate that VNN can handle a tensor that EXCEEDS RAM
    # Total RAM is around 17GB. Let's aim for a 32GB tensor (f32, so 8 billion elements)
    n = 8 * 1024 * 1024 * 1024 
    size_gb = (n * 4) / 1e9
    
    print(f"\n[!] Target Tensor Size: {size_gb:.1f} GB")
    print(f"[!] System RAM: ~17.0 GB")
    
    # 1. Attempt PyTorch Allocation (EXPECT FAILURE / CRASH)
    print("\n[1] Attempting PyTorch allocation (CPU)...")
    try:
        import torch as pt
        t_start = time.perf_counter()
        # This will likely raise a RuntimeError: [enforce fail at ... CPU allocator: out of memory]
        a_pt = pt.zeros(n, dtype=pt.float32)
        print("    SUCCESS (Unexpected! Is there swap?)")
    except Exception as e:
        print(f"    CAUGHT EXPECTED ERROR: {e}")
    
    # 2. VNN Allocation (SSD Streamed)
    print("\n[2] Attempting VNN SSD-Streamed allocation...")
    t_vnn_start = time.perf_counter()
    try:
        # This will automatically use SSD because size > safe_budget
        a_vnn = Tensor(None, shape=(n,), device='auto', dtype='float32')
        print(f"    VNN SUCCESS: Allocated {size_gb:.1f}GB on SSD.")
        
        # 3. Perform a huge operation
        print(f"\n[3] Performing VNN sum on {size_gb:.1f}GB tensor...")
        t_op_start = time.perf_counter()
        s = a_vnn.sum().item()
        t_op_end = time.perf_counter()
        
        print(f"    Sum Result: {s}")
        print(f"    Operation Time: {t_op_end - t_op_start:.2f}s")
        print(f"    Effective Throughput: {size_gb / (t_op_end - t_op_start):.1f} GB/s")
        
    except Exception as e:
        print(f"    VNN FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_dominance()
