import vulkannn_rusted as vnn
import numpy as np
import torch

def debug_fp16_vulkan():
    M, K, N = 128, 128, 128
    np.random.seed(42)
    a_np = np.random.rand(M, K).astype(np.float32)
    b_np = np.random.rand(K, N).astype(np.float32)
    
    a_vnn = vnn.Tensor(data=a_np, dtype=vnn.DataType.F16, device="vulkan")
    b_vnn = vnn.Tensor(data=b_np, dtype=vnn.DataType.F16, device="vulkan")
    c_vnn = (a_vnn @ b_vnn).to_numpy()
    
    a_torch = torch.from_numpy(a_np).to(torch.float16)
    b_torch = torch.from_numpy(b_np).to(torch.float16)
    c_torch = (a_torch @ b_torch).to(torch.float32).numpy()
    
    diff = np.abs(c_vnn - c_torch)
    print("Vulkan Max Diff:", diff.max())

def debug_fp16_hybrid():
    M, K, N = 128, 128, 128
    np.random.seed(42)
    a_np = np.random.rand(M, K).astype(np.float32)
    b_np = np.random.rand(K, N).astype(np.float32)
    
    a_vnn = vnn.Tensor(data=a_np, dtype=vnn.DataType.F16, device="hybrid")
    b_vnn = vnn.Tensor(data=b_np, dtype=vnn.DataType.F16, device="hybrid")
    c_vnn = (a_vnn @ b_vnn).to_numpy()
    
    a_torch = torch.from_numpy(a_np).to(torch.float16)
    b_torch = torch.from_numpy(b_np).to(torch.float16)
    c_torch = (a_torch @ b_torch).to(torch.float32).numpy()
    
    diff = np.abs(c_vnn - c_torch)
    print("Hybrid Max Diff:", diff.max())

debug_fp16_vulkan()
debug_fp16_hybrid()
