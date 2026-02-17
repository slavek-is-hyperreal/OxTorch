import os
import time
import numpy as np
import vulkan_nn_lib.torch_shim as torch
from vulkan_nn_lib.memory import MemoryManager

def benchmark_vulkan_tiling():
    # Target: 4GB Tensor (to exceed the 1GB VRAM detected on this system)
    N = 1_000_000_000 # ~4GB
    
    print(f"\n--- Heterogeneous Benchmark: Vulkan Tiling (4GB Tensor) ---")
    vram_info = MemoryManager.get_vram_info()
    print(f"System VRAM: {vram_info['Total']/1e9:.1f}GB total, {vram_info['Available']/1e9:.1f}GB available")
    
    # Force GPU Tiling
    print(f"\n[1/1] Initializing 4GB SSD Tensor and calculating (A * 2.0).sum() on GPU...")
    try:
        # Create SSD tensor (will be automatically offloaded due to size)
        a = torch.ones(N, device='ssd')
        a[0:1000] = 5.0 # Set some values to verify result
        
        t0 = time.perf_counter()
        # Ensure it uses Vulkan device for the streaming operation
        res = a.fused_sum(2.0, op='mul') # This should trigger SOE.elementwise_reduce with vulkan
        val = res.item()
        t_total = time.perf_counter() - t0
        
        # Expected: (1,000,000,000 - 1000) * 1.0 * 2.0 + 1000 * 5.0 * 2.0
        # = 999,999,000 * 2 + 10,000 = 1,999,998,000 + 10,000 = 2,000,008,000
        expected = (N - 1000) * 2.0 + 1000 * 10.0
        
        print(f"      Result: {val:.1f} (Expected: {expected:.1f})")
        print(f"      VNN (Vulkan Tiled) Time: {t_total:.2f}s")
        print(f"      Throughput: {(N * 4) / t_total / 1e6:.1f} MB/s")
        
        if abs(val - expected) < 1.0:
            print("\n      SUCCESS: Numerical parity confirmed on GPU tiles!")
        else:
            print(f"\n      FAILURE: Math mismatch! Diff: {abs(val-expected)}")
            
    except Exception as e:
        print(f"      Benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_vulkan_tiling()
