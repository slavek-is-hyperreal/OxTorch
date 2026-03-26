import torch
import torch.nn as nn
import oxtorch as vnn
import numpy as np
import time

def quant_activation(x):
    # symmetric per-token quantization
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale

def test_parity():
    # Shape for i5-3450 (SSE) and R7 260X (Vulkan)
    M = 1
    K = 4096
    N = 2560 # One layer dimension
    
    print(f"Creating test tensors: M={M}, K={K}, N={N}")
    
    # 1. Random ternary weights {-1, 0, 1}
    W_f32 = torch.randint(-1, 2, (N, K)).float()
    
    # 2. Random input activations
    X_f32 = torch.randn(M, K)
    X_q, scale_x = quant_activation(X_f32)
    
    # 3. Scale (per-channel/per-row)
    S_np = np.random.rand(N).astype(np.float32)
    S_vnn = vnn.Tensor(S_np, dtype="float32")
    
    # --- REFERENCE ---
    # Rescale X_q to float, multiply by W, multiply by S
    X_ref = X_q.float()
    W_ref = W_f32
    Ref_sum = torch.matmul(X_ref, W_ref.T)
    Ref_final = Ref_sum * torch.from_numpy(S_np) / scale_x
    
    # --- VNN BACKENDS ---
    # Pack weights for VNN (BitNet2 format)
    W_vnn = vnn.Tensor(W_f32.numpy(), dtype="int8").to_bitnet("bitnet2")
    X_vnn = vnn.Tensor(X_q.cpu().float().numpy(), dtype="int8")
    
    # A. CPU
    print("\n[CPU] Running inference...")
    X_vnn_cpu = X_vnn.to("cpu")
    W_vnn_cpu = W_vnn.to("cpu")
    S_vnn_cpu = S_vnn.to("cpu")
    
    start = time.time()
    res_vnn_cpu = X_vnn_cpu.bit_linear(W_vnn_cpu, S_vnn_cpu)
    cpu_time = (time.time() - start) * 1000
    res_cpu_f32 = torch.from_numpy(res_vnn_cpu.to_numpy()) / scale_x.cpu()
    
    # B. GPU (Vulkan)
    print("[Vulkan] Running inference...")
    X_vnn_vga = X_vnn.to("vga")
    W_vnn_vga = W_vnn.to("vga")
    S_vnn_vga = S_vnn.to("vga")
    
    start = time.time()
    res_vnn_vga = X_vnn_vga.bit_linear(W_vnn_vga, S_vnn_vga)
    vga_time = (time.time() - start) * 1000
    res_vga_f32 = res_vnn_vga.to_torch() / scale_x.cpu()
    
    # --- PARITY CHECK ---
    diff_cpu = torch.max(torch.abs(Ref_final - res_cpu_f32)).item()
    diff_vga = torch.max(torch.abs(Ref_final - res_vga_f32)).item()
    diff_cpu_vga = torch.max(torch.abs(res_cpu_f32 - res_vga_f32)).item()
    
    print(f"\nResults (Parity):")
    print(f"Max Diff (Ref vs CPU):    {diff_cpu:.6f}")
    print(f"Max Diff (Ref vs Vulkan): {diff_vga:.6f}")
    print(f"Max Diff (CPU vs Vulkan): {diff_cpu_vga:.6f}")
    
    print(f"\nPerformance (HPC Check):")
    print(f"CPU Time:    {cpu_time:.2f} ms")
    print(f"Vulkan Time: {vga_time:.2f} ms")
    
    if diff_cpu < 1e-4 and diff_vga < 1e-4:
        print("\n✅ PARITY PASSED!")
    else:
        print("\n❌ PARITY FAILED!")
        if diff_cpu > 1e-4: print("Hint: Check CPU bias correction.")
        if diff_vga > 1e-4: print("Hint: Check Vulkan bit-unpacking logic.")

if __name__ == "__main__":
    test_parity()
