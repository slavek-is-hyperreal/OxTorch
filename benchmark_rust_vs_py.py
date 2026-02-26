import os
import time
import numpy as np
import vulkannn_rusted as vnn_rust
import vulkan_nn_lib as vnn_py
from vulkan_nn_lib.tensor import Tensor as PyTensor

def benchmark_addition():
    print("--- Benchmark: Python+Taichi vs Rust+WGPU ---")
    size = 10_000_000 // 4 # ~10MB tensors
    
    # Setup data
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    
    # 1. Python + Taichi
    print("Initiating Taichi Engine...")
    t_py_a = PyTensor(a_np)
    t_py_b = PyTensor(b_np)
    
    # Warmup
    _ = t_py_a + t_py_b
    
    start_py = time.perf_counter()
    res_py = t_py_a + t_py_b
    end_py = time.perf_counter()
    time_py = (end_py - start_py) * 1000
    print(f"Taichi Addition Time: {time_py:.2f} ms")
    
    # 2. Rust + WGPU
    print("Initiating Rust Engine...")
    t_rust_a = vnn_rust.Tensor(a_np)
    t_rust_b = vnn_rust.Tensor(b_np)
    
    # Warmup
    _ = t_rust_a + t_rust_b
    
    start_rust = time.perf_counter()
    res_rust = t_rust_a + t_rust_b
    end_rust = time.perf_counter()
    time_rust = (end_rust - start_rust) * 1000
    print(f"Rust WGPU Addition Time: {time_rust:.2f} ms")
    
    print(f"\nRust is {time_py / time_rust:.2f}x faster on this operation!")
    
    # Verify parity
    np.testing.assert_allclose(res_py.to_numpy(), res_rust.to_numpy(), rtol=1e-5)
    print("✅ Mathematical Parity Confirmed!")

if __name__ == "__main__":
    benchmark_addition()
