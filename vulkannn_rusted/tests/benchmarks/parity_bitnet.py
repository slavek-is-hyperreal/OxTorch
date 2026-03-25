import time
import numpy as np
import vulkannn_rusted as vnn
import oxtorch as ox # Use OxTorch for idiomatic testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def reference_bit_linear(x_int8, w_trinary, scale, bias=None):
    """
    Reference BitNet b1.58 logic for INT8 inputs:
    y = (x_int8 * w_trinary^T) * scale + bias
    """
    y = torch.matmul(x_int8.to(torch.float32), w_trinary.to(torch.float32).t())
    # Broadast scale if it's a scalar or 1D
    y = y * scale
    if bias is not None:
        y = y + bias
    return y

def test_bit_linear_parity(M=1, K=4096, N=4096):
    print(f"--- BitNet Parity Test (M={M}, K={K}, N={N}) ---")
    
    if not HAS_TORCH:
        print("Skipping parity test (PyTorch not found).")
        return

    # 1. Prepare Data (PyTorch)
    # BitNet weights are {-1, 0, 1}
    w_pt = torch.randint(-1, 2, (N, K)).to(torch.float32)
    # ACTIVATIONS MUST BE INT8
    x_pt_int8 = torch.randint(-128, 128, (M, K), dtype=torch.int8)
    
    scale_val = 0.01
    # VNN kernel currently expects a per-channel scale (N,)
    scale_pt = torch.full((N,), scale_val).to(torch.float32)
    bias_pt = torch.randn((N,)).to(torch.float32)

    # Reference Output
    ref_y = reference_bit_linear(x_pt_int8, w_pt, scale_pt, bias_pt)

    # 2. Prepare VNN Tensors
    # VNN constructor requires float32 numpy array for data, even for Int8 target
    x_vnn = ox.Tensor(data=x_pt_int8.numpy().astype(np.float32), dtype=ox.int8, name="input")
    w_vnn_raw = ox.Tensor(data=w_pt.numpy().astype(np.float32), dtype=ox.int8, name="weight") # Pack from INT8
    w_vnn = w_vnn_raw.to_bitnet(ox.DataType.BitNet2)
    s_vnn = ox.Tensor(data=scale_pt.numpy().astype(np.float32), name="scale")
    b_vnn = ox.Tensor(data=bias_pt.numpy().astype(np.float32), name="bias")

    # 3. Execute VNN CPU Path
    x_vnn_cpu = x_vnn.to("cpu")
    w_vnn_cpu = w_vnn.to("cpu")
    s_vnn_cpu = s_vnn.to("cpu")
    b_vnn_cpu = b_vnn.to("cpu")
    
    # Instance method call as per new_op_tutorial.md
    _ = x_vnn_cpu.bit_linear(w_vnn_cpu, s_vnn_cpu, b_vnn_cpu) # Warmup
    
    start = time.perf_counter()
    y_vnn_cpu = x_vnn_cpu.bit_linear(w_vnn_cpu, s_vnn_cpu, b_vnn_cpu)
    cpu_time = (time.perf_counter() - start) * 1000

    # 4. Execute VNN Vulkan Path
    x_vnn_gpu = x_vnn.to("vulkan")
    w_vnn_gpu = w_vnn.to("vulkan")
    s_vnn_gpu = s_vnn.to("vulkan")
    b_vnn_gpu = b_vnn.to("vulkan")
    
    _ = x_vnn_gpu.bit_linear(w_vnn_gpu, s_vnn_gpu, b_vnn_gpu) # Warmup
    
    start = time.perf_counter()
    y_vnn_gpu = x_vnn_gpu.bit_linear(w_vnn_gpu, s_vnn_gpu, b_vnn_gpu)
    gpu_time = (time.perf_counter() - start) * 1000

    # 5. Compare Results
    y_cpu_np = y_vnn_cpu.to_numpy()
    y_gpu_np = y_vnn_gpu.to_numpy()
    ref_np = ref_y.numpy()

    mse_cpu = np.mean((y_cpu_np - ref_np)**2)
    mse_gpu = np.mean((y_gpu_np - ref_np)**2)

    print(f"CPU Time: {cpu_time:.2f} ms | MSE: {mse_cpu:.2e}")
    print(f"GPU Time: {gpu_time:.2f} ms | MSE: {mse_gpu:.2e}")
    
    if mse_cpu < 1e-4 and mse_gpu < 1e-4:
        print("SUCCESS: Parity Verified!")
    else:
        print("FAILURE: Parity Mismatch!")
        # Print first few elements for debugging
        print(f"Ref: {ref_np.flatten()[:5]}")
        print(f"CPU: {y_cpu_np.flatten()[:5]}")
        print(f"GPU: {y_gpu_np.flatten()[:5]}")

if __name__ == "__main__":
    test_bit_linear_parity(M=1, K=4096, N=4096)
