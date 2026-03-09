import torch
import numpy as np
import vulkannn_rusted as vnn

def test_bf16_conversion():
    print("=== BF16 CONVERSION DIAGNOSTIC ===")
    
    # Generate some values
    vals = torch.linspace(-10, 10, 20).float()
    print(f"Original F32: {vals.numpy()}")
    
    # PyTorch BF16
    t_bf16 = vals.bfloat16()
    t_f32_back = t_bf16.to(torch.float32).numpy()
    print(f"PyTorch BF16->F32: {t_f32_back}")
    
    # VNN BF16
    v_tensor = vnn.Tensor(vals.numpy(), dtype=vnn.DataType.BF16)
    v_f32_back = v_tensor.to_numpy().flatten()
    print(f"VNN BF16->F32:     {v_f32_back}")
    
    diff = np.abs(t_f32_back - v_f32_back)
    print(f"Max Diff: {np.max(diff)}")
    
    if np.max(diff) > 0:
        print("Mismatch found in basic conversion!")

def test_bf16_matmul_small():
    print("\n=== SMALL BF16 MATMUL DIAGNOSTIC ===")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).bfloat16()
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).bfloat16()
    
    res_pt = torch.mm(a, b).to(torch.float32).numpy()
    print(f"PyTorch Result:\n{res_pt}")
    
    v_a = vnn.Tensor(a.float().numpy(), dtype=vnn.DataType.BF16, device="cpu")
    v_b = vnn.Tensor(b.float().numpy(), dtype=vnn.DataType.BF16, device="cpu")
    res_vnn = (v_a @ v_b).to_numpy()
    print(f"VNN CPU Result:\n{res_vnn}")
    
    diff = np.abs(res_pt - res_vnn)
    print(f"Max Diff (CPU): {np.max(diff)}")

    v_a_v = vnn.Tensor(a.float().numpy(), dtype=vnn.DataType.BF16, device="vulkan")
    v_b_v = vnn.Tensor(b.float().numpy(), dtype=vnn.DataType.BF16, device="vulkan")
    res_vnn_v = (v_a_v @ v_b_v).to_numpy()
    print(f"VNN Vulkan Result:\n{res_vnn_v}")
    print(f"Max Diff (Vulkan): {np.max(np.abs(res_pt - res_vnn_v))}")

if __name__ == "__main__":
    test_bf16_conversion()
    test_bf16_matmul_small()
