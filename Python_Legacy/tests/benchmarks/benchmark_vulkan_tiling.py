import os
import time
import numpy as np
import Python_Legacy.vulkan_nn_lib.torch_shim as torch
from Python_Legacy.vulkan_nn_lib.memory import MemoryManager

def run_benchmark(mode, N):
    print(f"\n--- Testing Mode: {mode.upper()} ---")
    
    # Create SSD tensor (will be automatically offloaded due to size)
    # We set the device attribute to force VNN's mode selection
    a = torch.ones(N, device='ssd')
    # Use a trick to set the device property for the streaming engine
    a.device = mode
    
    a[0:1000] = 5.0 # Set some values to verify result
    
    t0 = time.perf_counter()
    # Perform fused operation: mul by 2.0 then sum
    res = a.fused_sum(2.0, op='mul')
    val = res.item()
    t_total = time.perf_counter() - t0
    
    # Expected: (N - 1000) * 1.0 * 2.0 + 1000 * 5.0 * 2.0 = 2N - 2000 + 10000 = 2N + 8000
    expected = (N - 1000) * 2.0 + 1000 * 10.0
    
    print(f"      Result: {val:.1f} (Expected: {expected:.1f})")
    print(f"      Time: {t_total:.2f}s | Throughput: {(N * 4) / t_total / 1e6:.1f} MB/s")
    
    diff = abs(val - expected)
    if diff < 0.1: # Expect exact parity with f64/double accumulators
        print(f"      SUCCESS: Precision confirmed!")
    else:
        print(f"      FAILURE: Math mismatch! Diff: {diff}")
    
    return t_total

def benchmark_heterogeneous():
    # Target: 4GB Tensor
    N = 1_000_000_000 # ~4GB
    
    print(f"====================================================")
    print(f"VNN Legacy Phase 6: Heterogeneous 'Monster' Benchmark")
    print(f"====================================================")
    
    vram_info = MemoryManager.get_vram_info()
    print(f"Hardware: {vram_info['Total']/1e9:.1f}GB VRAM ({vram_info['Available']/1e9:.1f}GB avail)")

    try:
        results = {}
        # Test 1: CPU Only
        # results['cpu'] = run_benchmark('cpu', N)
        
        # Test 2: Hybrid (CPU + GPU)
        results['hybrid'] = run_benchmark('hybrid', N)
        
        # Test 3: Vulkan Only (GPU)
        results['vulkan'] = run_benchmark('vulkan', N)
        
        print("\nSummary Results:")
        for mode, t in results.items():
            print(f"  {mode.upper()}: {t:.2f}s ({(N*4)/t/1e6:.1f} MB/s)")
            
    except Exception as e:
        print(f"\n      Benchmark CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_heterogeneous()
