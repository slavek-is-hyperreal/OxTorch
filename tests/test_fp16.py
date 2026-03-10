import vulkannn_rusted as vnn
import numpy as np
import torch

def test_fp16_basic():
    print("--- Testing FP16 Basic ---")
    # 1. Create FP16 Tensor
    shape = (2, 3)
    data = np.random.rand(*shape).astype(np.float32)
    t = vnn.Tensor(data=data, dtype=vnn.DataType.F16)
    print(f"Tensor: {t}")
    print(f"Data type: {t.dtype}")
    
    # 2. To Numpy
    out_np = t.to_numpy()
    print(f"Output NP shape: {out_np.shape}, dtype: {out_np.dtype}")
    diff = np.abs(data - out_np).max()
    print(f"Max diff from original (F32->F16->F32): {diff}")
    
    # 3. Add
    t2 = vnn.Tensor(data=np.ones(shape, dtype=np.float32), dtype=vnn.DataType.F16)
    t_sum = t + t2
    sum_np = t_sum.to_numpy()
    expected = data + 1.0
    diff_sum = np.abs(sum_np - expected).max()
    print(f"Add max diff: {diff_sum}")

def test_fp16_matmul():
    print("\n--- Testing FP16 MatMul ---")
    M, K, N = 128, 128, 128
    a_np = np.random.rand(M, K).astype(np.float32)
    b_np = np.random.rand(K, N).astype(np.float32)
    
    a_vnn = vnn.Tensor(data=a_np, dtype=vnn.DataType.F16)
    b_vnn = vnn.Tensor(data=b_np, dtype=vnn.DataType.F16)
    
    c_vnn = a_vnn @ b_vnn
    c_np = c_vnn.to_numpy()
    
    # torch reference
    a_torch = torch.from_numpy(a_np).to(torch.float16)
    b_torch = torch.from_numpy(b_np).to(torch.float16)
    c_torch = (a_torch @ b_torch).to(torch.float32).numpy()
    
    diff = np.abs(c_np - c_torch).max()
    print(f"MatMul max diff vs Torch FP16: {diff}")
    if diff < 5e-2:
        print("MatMul FP16 SUCCESS")
    else:
        print("MatMul FP16 FAILURE - high difference")

if __name__ == "__main__":
    test_fp16_basic()
    test_fp16_matmul()
