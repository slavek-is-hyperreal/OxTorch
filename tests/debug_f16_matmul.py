import vulkannn_rusted as vnn
import numpy as np
import torch

def debug_fp16():
    M, K, N = 4, 4, 4
    np.random.seed(42)
    a_np = np.random.rand(M, K).astype(np.float32)
    b_np = np.random.rand(K, N).astype(np.float32)
    
    a_vnn = vnn.Tensor(data=a_np, dtype=vnn.DataType.F16)
    b_vnn = vnn.Tensor(data=b_np, dtype=vnn.DataType.F16)
    c_vnn = (a_vnn @ b_vnn).to_numpy()
    
    a_torch = torch.from_numpy(a_np).to(torch.float16)
    b_torch = torch.from_numpy(b_np).to(torch.float16)
    c_torch = (a_torch @ b_torch).to(torch.float32).numpy()
    
    print("A (F32 orig):\\n", a_np)
    print("A (Torch F16 -> F32):\\n", a_torch.to(torch.float32).numpy())
    print("A (VNN F16 -> F32):\\n", a_vnn.to_numpy())
    print("\\nC (VNN output):\\n", c_vnn)
    print("C (Torch output):\\n", c_torch)
    print("\\nDiff:\\n", np.abs(c_vnn - c_torch))

debug_fp16()
