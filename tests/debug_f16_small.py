import torch
import numpy as np
import vulkannn_rusted as vnn
from vulkannn_rusted import Tensor

def test_small_matmul_f16():
    print("--- Small F16 MatMul Parity Test ---")
    m, k, n = 4, 3, 2
    a_np = np.arange(m * k).reshape(m, k).astype(np.float32)
    b_np = np.arange(k * n).reshape(k, n).astype(np.float32)
    
    print("Matrix A (4x3):\n", a_np)
    print("Matrix B (3x2):\n", b_np)
    
    a_torch = torch.from_numpy(a_np).to(torch.float16)
    b_torch = torch.from_numpy(b_np).to(torch.float16)
    res_torch = torch.matmul(a_torch, b_torch).to(torch.float32).numpy()
    
    a_vnn = Tensor(data=a_np, dtype=vnn.DataType.F16, device="cpu", name="A")
    b_vnn = Tensor(data=b_np, dtype=vnn.DataType.F16, device="cpu", name="B")
    res_vnn_tensor = a_vnn @ b_vnn
    res_vnn = res_vnn_tensor.to_numpy()
    
    print("\nPyTorch Result:\n", res_torch)
    print("\nVNN Result:\n", res_vnn)
    
    diff = np.abs(res_torch - res_vnn)
    print("\nMax Diff:", np.max(diff))
    if np.max(diff) < 1e-2:
        print("✅ Parity OK")
    else:
        print("❌ Parity FAILED")

if __name__ == "__main__":
    test_small_matmul_f16()
