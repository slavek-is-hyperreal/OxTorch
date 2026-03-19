import torch
import vulkannn_rusted as vnn
import numpy as np

def test_f32_vulkan_matmul(M, K, N):
    print(f"Testing F32 Vulkan MatMul Parity ({M}x{K}x{N})...")
    
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    
    # PyTorch Reference
    a_pt = torch.from_numpy(a_np)
    b_pt = torch.from_numpy(b_np)
    res_pt = torch.matmul(a_pt, b_pt)
    
    # VNN Vulkan
    a_vnn = vnn.Tensor(data=a_np, dtype=vnn.DataType.F32, device="vulkan")
    b_vnn = vnn.Tensor(data=b_np, dtype=vnn.DataType.F32, device="vulkan")
    
    # VNN MatMul
    res_vnn = a_vnn @ b_vnn
    res_vnn_np = res_vnn.to_numpy()
    
    diff = np.abs(res_pt.numpy() - res_vnn_np)
    max_diff = np.max(diff)
    print(f"Max Diff: {max_diff}")
    
    if max_diff < 1e-3:
        print(f"✅ F32 Vulkan MatMul ({M}x{K}x{N}): PASS")
    else:
        print(f"❌ F32 Vulkan MatMul ({M}x{K}x{N}): FAIL")
        print("Sample Reference:", res_pt.numpy()[0, :5])
        print("Sample VNN:", res_vun_np[0, :5] if 'res_vun_np' in locals() else res_vnn_np[0, :5])

if __name__ == "__main__":
    test_f32_vulkan_matmul(128, 128, 128)
    test_f32_vulkan_matmul(129, 129, 129)
    test_f32_vulkan_matmul(2048, 2048, 2048)
