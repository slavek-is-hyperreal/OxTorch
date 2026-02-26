import os
import time
import numpy as np
import vulkannn_rusted as vnn_rust
import vulkan_nn_lib as vnn_py
from vulkan_nn_lib.tensor import Tensor as PyTensor

def create_temp_file(filename, size_in_bytes):
    # Create file with random floats
    num_floats = size_in_bytes // 4
    # To avoid memory spike, write in chunks
    chunk_size = 10_000_000
    with open(filename, 'wb') as f:
        for _ in range(0, num_floats, chunk_size):
            chunk = np.random.rand(min(chunk_size, num_floats)).astype(np.float32)
            f.write(chunk.tobytes())
            num_floats -= chunk_size

def benchmark_op(op_name, size_mb, a_np, b_np=None):
    results = {}
    
    # --- Python Taichi ---
    try:
        t_py_a = PyTensor(a_np)
        if b_np is not None:
             t_py_b = PyTensor(b_np)
             
        # Warmup
        if op_name == "Add": _ = t_py_a + t_py_b
        elif op_name == "MatMul": _ = t_py_a @ t_py_b
        elif op_name == "ReLU": _ = t_py_a.relu()
        elif op_name == "Sigmoid": _ = t_py_a.sigmoid()
        
        start = time.perf_counter()
        if op_name == "Add": res_py = t_py_a + t_py_b
        elif op_name == "MatMul": res_py = t_py_a @ t_py_b
        elif op_name == "ReLU": res_py = t_py_a.relu()
        elif op_name == "Sigmoid": res_py = t_py_a.sigmoid()
        # Wait for taichi sync if needed
        res_np_py = res_py.to_numpy() 
        time_py = (time.perf_counter() - start) * 1000
        results['python_ms'] = time_py
    except Exception as e:
        results['python_ms'] = -1
        print(f"Python failed {op_name}: {e}")

    # --- Rust WGPU ---
    try:
        t_rust_a = vnn_rust.Tensor(a_np)
        if b_np is not None:
             t_rust_b = vnn_rust.Tensor(b_np)
             
        # Warmup
        if op_name == "Add": _ = t_rust_a + t_rust_b
        elif op_name == "MatMul": _ = t_rust_a @ t_rust_b
        elif op_name == "ReLU": _ = t_rust_a.relu()
        elif op_name == "Sigmoid": _ = t_rust_a.sigmoid()
        
        start = time.perf_counter()
        if op_name == "Add": res_rust = t_rust_a + t_rust_b
        elif op_name == "MatMul": res_rust = t_rust_a @ t_rust_b
        elif op_name == "ReLU": res_rust = t_rust_a.relu()
        elif op_name == "Sigmoid": res_rust = t_rust_a.sigmoid()
        res_np_rust = res_rust.to_numpy()
        time_rust = (time.perf_counter() - start) * 1000
        results['rust_ms'] = time_rust
        
        # Verify Parity
        if results.get('python_ms', -1) != -1:
            np.testing.assert_allclose(res_np_py, res_np_rust, rtol=1e-4, atol=1e-4)
            results['parity'] = "✅"
        else:
            results['parity'] = "N/A"
            
    except Exception as e:
        results['rust_ms'] = -1
        results['parity'] = "❌"
        print(f"Rust failed {op_name}: {e}")

    return results

def run_benchmarks():
    print("| Operation | Size | Python (Taichi) | Rust (WGPU) | Speedup | Parity |")
    print("|---|---|---|---|---|---|")
    
    sizes_mb = [10, 100] # MB
    
    for size in sizes_mb:
        num_floats = (size * 1024 * 1024) // 4
        
        # 1D arrays
        a_1d = np.random.rand(num_floats).astype(np.float32)
        b_1d = np.random.rand(num_floats).astype(np.float32)
        
        for op in ["Add", "ReLU", "Sigmoid"]:
            res = benchmark_op(op, size, a_1d, b_1d if op == "Add" else None)
            py_t = f"{res['python_ms']:.1f}ms" if res['python_ms'] != -1 else "OOM/Err"
            ru_t = f"{res['rust_ms']:.1f}ms" if res['rust_ms'] != -1 else "OOM/Err"
            speedup = f"{res['python_ms']/res['rust_ms']:.2f}x" if res['rust_ms'] > 0 and res['python_ms'] > 0 else "N/A"
            print(f"| {op} | {size}MB | {py_t} | {ru_t} | {speedup} | {res.get('parity', '-')} |")
            
        # 2D arrays for MatMul (approximate size)
        dim = int(np.sqrt(num_floats))
        a_2d = np.random.rand(dim, dim).astype(np.float32)
        b_2d = np.random.rand(dim, dim).astype(np.float32)
        
        res = benchmark_op("MatMul", size, a_2d, b_2d)
        py_t = f"{res['python_ms']:.1f}ms" if res['python_ms'] != -1 else "OOM/Err"
        ru_t = f"{res['rust_ms']:.1f}ms" if res['rust_ms'] != -1 else "OOM/Err"
        speedup = f"{res['python_ms']/res['rust_ms']:.2f}x" if res['rust_ms'] > 0 and res['python_ms'] > 0 else "N/A"
        print(f"| MatMul | {dim}x{dim} | {py_t} | {ru_t} | {speedup} | {res.get('parity', '-')} |")

if __name__ == "__main__":
    run_benchmarks()
