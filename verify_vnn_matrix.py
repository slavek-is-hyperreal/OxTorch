import os
import numpy as np
import time
import torch as pt
import vulkan_nn_lib.torch_shim as vnn
from vulkan_nn_lib.tensor import Tensor
from vulkan_nn_lib.memory import MemoryManager

def format_speed(bytes_per_sec):
    if bytes_per_sec > 1e9: return f"{bytes_per_sec/1e9:.2f} GB/s"
    return f"{bytes_per_sec/1e6:.1f} MB/s"

def run_bench(name, size_bytes, mode, op_type='sum'):
    n = int(size_bytes // 4) # float32
    print(f"\n>>> TEST: {name} | Size: {size_bytes/1e6:.1f}MB | Mode: {mode} | Op: {op_type}")
    
    # Standardize data generator for parity, but skip for MONSTER scale
    a_np = None
    if size_bytes < 8e9:
        a_np = np.random.randn(n).astype(np.float32)
    
    # 1. Baseline (Torch)
    torch_time = 0
    if size_bytes < 8e9: # Avoid Torch OOM
        try:
            a_pt = pt.from_numpy(a_np)
            t0 = time.perf_counter()
            if op_type == 'sum':
                res_pt = float(pt.sum(a_pt.to(pt.float64)).item())
            else:
                b_pt = pt.from_numpy(a_np.copy()) # Use copy for add parity
                res_pt_t = a_pt + b_pt
                res_pt = res_pt_t.numpy()
            torch_time = time.perf_counter() - t0
            print(f"    Torch Time: {torch_time:.4f}s | {format_speed(size_bytes/torch_time)}")
        except Exception as e:
            print(f"    Torch Failed/Skipped: {e}")
            torch_time = None
    else:
        print("    Torch Skipped (Monster Scale)")
        torch_time = None

    # 2. VNN
    try:
        if size_bytes < 8e9:
            a_vnn = vnn.tensor(a_np.copy(), device='cpu')
        else:
            # Monster scale: Use .ones() to have non-zero sum
            print(f"    [Factory] Initializing {size_bytes/1e9:.1f}GB ones tensor on SSD...")
            a_vnn = vnn.ones(n, device='ssd')
            
        # Manually override device metadata for testing dispatch
        if mode != 'auto' and mode != 'ssd':
            # This is a bit hacky, but for benchmarking the PATHS we force the label
            # Real 'vulkan' move would require a .cuda() call equivalent
            a_vnn.device = mode

        t0 = time.perf_counter()
        if op_type == 'sum':
            res_vnn_t = a_vnn.sum()
            res_vnn = res_vnn_t.item()
        else:
            b_vnn = vnn.tensor(a_np.copy(), device='cpu')
            if mode != 'auto': b_vnn.device = mode
            res_vnn_t = a_vnn + b_vnn
            res_vnn = res_vnn_t.numpy()
        
        vnn_time = time.perf_counter() - t0
        print(f"    VNN Time:   {vnn_time:.4f}s | {format_speed(size_bytes/vnn_time)}")
        
        if torch_time:
            ratio = vnn_time / torch_time
            print(f"    Ratio vs Torch: {ratio:.2f}x")
            
        # Numerical Check
        if torch_time and op_type == 'sum':
            diff = abs(res_vnn - res_pt)
            print(f"    Numerical Diff: {diff:.2e} (Abs error)")
            if diff > 1.0: print("    [!] WARNING: High error!")

    except Exception as e:
        print(f"    VNN FAILED: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== VNN COMPREHENSIVE VERIFICATION MATRIX (Standardized Data) ===")
    
    # Setup SSD Cache
    Tensor.setup_ssd_storage("/vectorlegis_ssd_pool/vnn_cache")
    
    # 1. 128MB (RAM) - CPU Baseline
    run_bench("Small-RAM-CPU", 128*1024*1024, 'cpu', 'sum')
    
    # 2. 128MB (RAM) - Vulkan (Simulated Move)
    # Note: For real Vulkan perf, we'd need to actually move data.
    # Setting .device='vulkan' forces it to use kernels but it will fail if data is numpy.
    # I'll fix the engine to auto-move if requested.
    
    # 3. 4GB (Large-RAM) - CPU (Should now hit Fast Path < 1.5x)
    run_bench("Large-RAM-CPU", 4*1024*1024*1024, 'cpu', 'sum')
    
    # 4. 34GB (Monster-SSD) - Hybrid
    run_bench("Monster-SSD-Hybrid", 34.4*1024*1024*1024, 'hybrid', 'sum')

if __name__ == "__main__":
    main()
