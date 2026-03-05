import torch
import numpy as np
import vulkannn_rusted as vnn
from vulkannn_rusted import Tensor

def test_small_matmul_prec(dtype_str):
    print(f"\n--- Small {dtype_str} MatMul Parity Test ---")
    if dtype_str == "F16":
        vnn_dtype = vnn.DataType.F16
        torch_dtype = torch.float16
    elif dtype_str == "BF16":
        vnn_dtype = vnn.DataType.BF16
        torch_dtype = torch.bfloat16
    else:
        vnn_dtype = vnn.DataType.F32
        torch_dtype = torch.float32

    m, k, n = 4, 3, 2
    a_np = np.arange(m * k).reshape(m, k).astype(np.float32)
    b_np = np.arange(k * n).reshape(k, n).astype(np.float32)
    
    # print("Matrix A (4x3):\n", a_np)
    # print("Matrix B (3x2):\n", b_np)
    
    a_torch = torch.from_numpy(a_np).to(torch_dtype)
    b_torch = torch.from_numpy(b_np).to(torch_dtype)
    res_torch = torch.matmul(a_torch, b_torch).to(torch.float32).numpy()
    
    a_vnn = Tensor(data=a_np, dtype=vnn_dtype, device="cpu", name="A")
    b_vnn = Tensor(data=b_np, dtype=vnn_dtype, device="cpu", name="B")
    res_vnn_tensor = a_vnn @ b_vnn
    res_vnn = res_vnn_tensor.to_numpy()
    
    print("PyTorch Result:\n", res_torch)
    print("VNN Result:\n", res_vnn)
    
    diff = np.abs(res_torch - res_vnn)
    max_diff = np.max(diff)
    print(f"Max Diff ({dtype_str}):", max_diff)
    if max_diff < 1e-2:
        print(f"✅ {dtype_str} Parity OK")
        return True
    else:
        print(f"❌ {dtype_str} Parity FAILED")
        return False

if __name__ == "__main__":
    ok_f16 = test_small_matmul_prec("F16")
    ok_bf16 = test_small_matmul_prec("BF16")
    
    if ok_f16 and ok_bf16:
        print("\n🎉 ALL SMALL PARITY TESTS PASSED")
    else:
        print("\n🚨 SOME PARITY TESTS FAILED")
