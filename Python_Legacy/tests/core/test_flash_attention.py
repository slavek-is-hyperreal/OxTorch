import numpy as np
import Python_Legacy.vulkan_nn_lib as vnn
from Python_Legacy.vulkan_nn_lib.tensor import Tensor
import math
import torch
import torch.nn.functional as F
import taichi as ti

def test_flash_attention_math():
    ti.init(arch=ti.vulkan)
    print("--- Testing FlashAttention vs PyTorch ---")
    
    seq_len = 128
    head_dim = 64
    batch_size = 2
    num_heads = 4
    
    # Q,K,V original shape: [B, L, H, D]
    q_np = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
    k_np = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
    v_np = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
    
    # 1. PyTorch baseline expects: [B, H, L, D]
    pt_q = torch.from_numpy(q_np).permute(0, 2, 1, 3)
    pt_k = torch.from_numpy(k_np).permute(0, 2, 1, 3)
    pt_v = torch.from_numpy(v_np).permute(0, 2, 1, 3)
    
    out_expected = F.scaled_dot_product_attention(pt_q, pt_k, pt_v).permute(0, 2, 1, 3).numpy()
    
    # 2. VNN FlashAttention
    q_vnn = Tensor(q_np, device='vulkan')
    k_vnn = Tensor(k_np, device='vulkan')
    v_vnn = Tensor(v_np, device='vulkan')
    out_vnn = Tensor(None, shape=(batch_size, seq_len, num_heads, head_dim), device='vulkan')
    import Python_Legacy.vulkan_nn_lib.kernels as K
    
    K.k_flash_attention_vulkan(q_vnn.arr, k_vnn.arr, v_vnn.arr, out_vnn.arr, batch_size, num_heads, seq_len, head_dim, 1.0 / math.sqrt(head_dim))
    ti.sync()
    
    out_actual = out_vnn.to_numpy()
    
    print(f"Max Diff: {np.max(np.abs(out_actual - out_expected))}")
    assert np.allclose(out_actual, out_expected, atol=1e-4)
    print("Mathematical baseline established AND Parity Verified!")

if __name__ == "__main__":
    test_flash_attention_math()
