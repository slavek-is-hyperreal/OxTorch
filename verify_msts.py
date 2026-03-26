import torch
import numpy as np
import vulkannn_rusted as vnn
import os
import time

def test_msts_paths():
    print("--- Verifying MSTS Logic ---")
    # Path A: Small (Direct) - 10KB
    print("\n[Path A] Testing Small Tensor (Direct)...")
    data_small = np.ascontiguousarray(np.random.randn(2500).astype(np.float32))
    t_small = vnn.Tensor(data=data_small, name="small")
    t_small_ssd = t_small.save_ssd("small.ssd")
    res_small = t_small_ssd.unary_op_ssd("relu", 0.0, 0.0)
    data = res_small.load_to_f32_vec_msts()
    print(f"Path A OK. Sample: {data[:5]}")

    # Path B: Medium (Single-thread) - 2MB
    print("\n[Path B] Testing Medium Tensor (Single-thread)...")
    data_med = np.ascontiguousarray(np.random.randn(512*1024).astype(np.float32))
    t_med = vnn.Tensor(data=data_med, name="med")
    t_med_ssd = t_med.save_ssd("med.ssd")
    res_med = t_med_ssd.unary_op_ssd("relu", 0.0, 0.0)
    data = res_med.load_to_f32_vec_msts()
    print(f"Path B OK. Size: {len(data)}")

    # Path C: Large (Full) - 200MB
    print("\n[Path C] Testing Large Tensor (Full Multi-worker)...")
    size = 50 * 1024 * 1024 # 50M floats = 200MB
    data_large = np.ascontiguousarray(np.random.randn(size).astype(np.float32))
    t_large = vnn.Tensor(data=data_large, name="large")
    t_large_ssd = t_large.save_ssd("large.ssd")
    start = time.time()
    res_large = t_large_ssd.unary_op_ssd("relu", 0.0, 0.0)
    end = time.time()
    print(f"Path C OK. Time: {end-start:.4f}s")
    
    # Cleanup
    for f in ["small.ssd", "med.ssd", "large.ssd", "small_relu.ssd", "med_relu.ssd", "large_relu.ssd"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    test_msts_paths()
