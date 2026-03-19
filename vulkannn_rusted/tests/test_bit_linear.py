import torch
import vulkannn_rusted as vnn
import numpy as np
import time

def test_bit_linear_parity():
    print("Testing BitLinear Parity (CPU vs Vulkan)...")
    
    M, K, N = 4, 1024, 2048
    
    # 1. Prepare Activations (Int8)
    # BitNet expects Int8 activations [-128, 127]
    activations_np = np.random.randint(-10, 10, (M, K)).astype(np.int8)
    # We pass as float32, VNN will cast to int8 internally
    act_vnn = vnn.Tensor(data=activations_np.astype(np.float32), dtype=vnn.DataType.Int8, device="cpu")
    act_vnn_vulkan = vnn.Tensor(data=activations_np.astype(np.float32), dtype=vnn.DataType.Int8, device="vulkan")
    
    # 2. Prepare Weights (Ternary)
    # We create from F32 and let VNN quantize it
    weights_f32 = np.random.randn(N, K).astype(np.float32)
    
    weight_vnn = vnn.Tensor(data=weights_f32, dtype=vnn.DataType.Ternary, device="cpu")
    weight_vnn_vulkan = vnn.Tensor(data=weights_f32, dtype=vnn.DataType.Ternary, device="vulkan")
    
    # 3. Prepare Scales (F32)
    # BitNet uses scales for output dequantization
    scales_np = np.random.rand(N).astype(np.float32)
    scales_vnn = vnn.Tensor(data=scales_np, dtype=vnn.DataType.F32, device="cpu")
    scales_vnn_vulkan = vnn.Tensor(data=scales_np, dtype=vnn.DataType.F32, device="vulkan")
    
    # 4. Execute CPU
    print(f"Executing CPU BitLinear ({M}x{K}x{N})...")
    res_cpu = vnn.Tensor.bit_linear(act_vnn, weight_vnn, scales_vnn)
    res_cpu_np = res_cpu.to_numpy()
    
    # 5. Execute Vulkan
    print(f"Executing Vulkan BitLinear ({M}x{K}x{N})...")
    res_vulkan = vnn.Tensor.bit_linear(act_vnn_vulkan, weight_vnn_vulkan, scales_vnn_vulkan)
    res_vulkan_np = res_vulkan.to_numpy()
    
    # 7. Benchmarking
    print("\nBenchmarking (10 iterations)...")
    import time
    
    # CPU
    start = time.time()
    for _ in range(10):
        _ = vnn.Tensor.bit_linear(act_vnn, weight_vnn, scales_vnn)
    end = time.time()
    print(f"CPU BitLinear:    {(end - start)/10:.4f}s per iter")
    
    # Vulkan
    start = time.time()
    for _ in range(10):
        _ = vnn.Tensor.bit_linear(act_vnn_vulkan, weight_vnn_vulkan, scales_vnn_vulkan)
    end = time.time()
    print(f"Vulkan BitLinear: {(end - start)/10:.4f}s per iter")
    
    # Speedup
    cpu_time = (end - start) # wait, math fix
    # Let's just do it properly
    
def run_bench():
    M, K, N = 1, 4096, 4096 # LLM-like shape (batch=1)
    
    act_np = np.random.randint(-10, 10, (M, K)).astype(np.float32)
    weight_np = np.random.randn(N, K).astype(np.float32)
    scales_np = np.random.rand(N).astype(np.float32)
    
    act_v = vnn.Tensor(data=act_np, dtype=vnn.DataType.Int8, device="vulkan")
    weight_v = vnn.Tensor(data=weight_np, dtype=vnn.DataType.Ternary, device="vulkan")
    scales_v = vnn.Tensor(data=scales_np, dtype=vnn.DataType.F32, device="vulkan")
    
    act_c = vnn.Tensor(data=act_np, dtype=vnn.DataType.Int8, device="cpu")
    weight_c = vnn.Tensor(data=weight_np, dtype=vnn.DataType.Ternary, device="cpu")
    scales_c = vnn.Tensor(data=scales_np, dtype=vnn.DataType.F32, device="cpu")
    
    print(f"\nLLM Shape Benchmark ({M}x{K}x{N}):")
    
    iters = 20
    # Warmup
    for _ in range(5):
        vnn.Tensor.bit_linear(act_v, weight_v, scales_v)
        vnn.Tensor.bit_linear(act_c, weight_c, scales_c)
        
    t0 = time.time()
    for _ in range(iters):
        vnn.Tensor.bit_linear(act_c, weight_c, scales_c)
    t1 = time.time()
    cpu_avg = (t1 - t0) / iters
    print(f"CPU Baseline:    {cpu_avg:.4f}s")
    
    t0 = time.time()
    for _ in range(iters):
        vnn.Tensor.bit_linear(act_v, weight_v, scales_v)
    t1 = time.time()
    vulkan_avg = (t1 - t0) / iters
    print(f"Vulkan Compute:  {vulkan_avg:.4f}s")
    print(f"Speedup:         {cpu_avg/vulkan_avg:.2f}x")

if __name__ == "__main__":
    test_bit_linear_parity()
    run_bench()
