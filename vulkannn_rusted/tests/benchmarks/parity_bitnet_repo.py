import torch
import numpy as np
import oxtorch as vnn
from safetensors.torch import load_file
import os

# Logic from BitNet_repo/gpu/model.py
def quant_input(x):
    s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_ (min=1e-5)
    # The reference is (X * s).round().clamp() / s
    # But VNN takes Int8 and a dequant scale.
    # So we pass (X * s).round() as Int8, and (1.0 / s) as dequant scale.
    x_q = (x * s).round().clamp(-128, 127).to(torch.int8)
    return x_q, 1.0 / s

# Logic from BitNet_repo/gpu/convert_checkpoint.py
def quant_weight(w):
    s = 1.0 / w.abs().mean().clamp_(min=1e-5)
    w_q = (w * s).round().clamp(-1, 1)
    return w_q, 1.0 / s

def pack_weights_bitnet2(w_q):
    """
    Packs ternary weights (-1, 0, 1) into uint8 (0, 1, 2)
    Maps: -1 -> 0, 0 -> 1, 1 -> 2
    Order: High bits first (Shifted-Sum compatible)
    """
    w_u = (w_q + 1).to(torch.uint8)
    N, K = w_u.shape
    packed = torch.zeros((N, K // 4), dtype=torch.uint8)
    for i in range(4):
        packed |= (w_u[:, i::4] << (2 * (3 - i)))
    return packed

def test_bitnet_repo_parity(M=1, K=2560, N=6912, device="cpu"):
    print(f"\n--- BitNet_repo Parity Test ({device.upper()}) ---")
    print(f"Shape: M={M}, K={K}, N={N}")

    # 1. Create random input and weights
    x = torch.randn((M, K))
    w = torch.randn((N, K))

    # 2. Reference Quantization (BitNet_repo logic)
    x_q, x_scale = quant_input(x)
    w_q, w_scale = quant_weight(w)

    # 3. Reference Forward Pass
    # BitNet_repo: out = (X_q @ W_q.T) * (w_scale / s_x)
    # Wait! x_scale here is 1/s, so (w_scale * x_scale)
    ref = (x_q.float() @ w_q.float().T) * (w_scale * x_scale)

    # 4. Preparing VNN
    # VNN BitLinear takes: x (Int8), weight (BitNet2 packed), scale (N,)
    vnn_w_scale = torch.full((N,), w_scale.item())
    vnn_total_scale = vnn_w_scale * x_scale.squeeze()

    # Weight packing: Use engine's native to_bitnet to ensure correct SWAR layout
    w_vnn_int8 = vnn.Tensor(w_q.numpy().astype(np.float32), dtype="int8")
    w_vnn = w_vnn_int8.to_bitnet("bitnet2").to(device)

    # Input and scale Tensors
    x_vnn = vnn.Tensor(x_q.numpy().astype(np.float32), dtype="int8").to(device)
    s_vnn = vnn.Tensor(vnn_total_scale.numpy().astype(np.float32), dtype="float32").to(device)

    # 5. Run VNN
    out_vnn = x_vnn.bit_linear(w_vnn, s_vnn)
    
    # Sync if GPU
    if device == "vga" or device == "vulkan":
        out_vnn = out_vnn.to("cpu")

    out_vnn_pt = torch.from_numpy(out_vnn.numpy())

    # 6. Compare
    diff = (ref - out_vnn_pt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max Diff: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("PASS \u2705")
    else:
        print("FAIL \u274c")

if __name__ == "__main__":
    # Test on CPU
    test_bitnet_repo_parity(device="cpu")
    # Test on GPU
    try:
        test_bitnet_repo_parity(device="vga")
    except Exception as e:
        print(f"GPU Test Skipped or Failed: {e}")
