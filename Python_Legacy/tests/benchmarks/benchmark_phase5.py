import Python_Legacy.vulkan_nn_lib.torch_shim as torch
import numpy as np
import time
import os

def benchmark_fusion():
    print("\n--- Phase 5: Operator Fusion & I/O Reduction Benchmark ---")
    
    # 2GB Tensor
    N = 500_000_000 
    print(f"Initializing {N*4/1e9:.1f}GB tensor on SSD...")
    a = torch.randn(N, device='ssd')
    b = 2.0
    
    # 1. Standard (creates intermediate SSD tensor)
    print("\n[Method 1] Standard: (a * 2.0).sum()")
    # We clear cache by assuming large enough N or just running twice
    t0 = time.perf_counter()
    res1 = (a * b).sum()
    print(f"Standard Result: {res1.item():.2f}")
    t1 = time.perf_counter()
    print(f"Standard Time: {t1-t0:.4f}s")
    
    # 2. Fused (no intermediate write)
    print("\n[Method 2] Fused: a.fused_sum(2.0, op='mul')")
    t0 = time.perf_counter()
    res2 = a.fused_sum(b, op='mul')
    print(f"Fused Result: {res2.item():.2f}")
    t1 = time.perf_counter()
    print(f"Fused Time: {t1-t0:.4f}s")
    
    print("\n--- Benchmark Complete ---")

if __name__ == "__main__":
    benchmark_fusion()
