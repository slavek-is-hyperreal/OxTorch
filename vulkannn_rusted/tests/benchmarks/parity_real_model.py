import torch
import torch.nn.functional as F
import numpy as np
import oxtorch as vnn
from safetensors.torch import load_file
import time
import os

def unpack_weights_hf(packed, out_features, in_features, dtype=torch.float32):
    # Unpacks weights from HF format (packed along out_features)
    # packed shape: (out_features // 4, in_features)
    # returns: (out_features, in_features)
    row_dim = out_features // 4
    unpacked = torch.zeros((out_features, in_features), dtype=torch.uint8)
    for i in range(4):
        start = i * row_dim
        end = start + row_dim
        mask = 3 << (2 * i)
        unpacked[start:end, :] = (packed & mask) >> (2 * i)
    return unpacked.to(dtype) - 1

def pack_weights_vnn(unpacked):
    # Packs weights for VNN (packed along in_features / K)
    # unpacked shape: (N, K)
    # returns: (N, (K+3)//4) as uint8
    N, K = unpacked.shape
    packed_K = (K + 3) // 4
    packed = np.zeros((N, packed_K), dtype=np.uint8)
    
    # Ternary values {-1, 0, 1} -> {0, 1, 2}
    u = (unpacked.numpy() + 1).astype(np.uint8)
    
    for k in range(K):
        byte_idx = k // 4
        bit_shift = (k % 4) * 2
        packed[:, byte_idx] |= (u[:, k] << bit_shift)
    
    return packed

def test_real_layer_parity():
    model_path = "/my_data/gaussian_room/models/bitnet-2B-ternary/model.safetensors"
    print(f"Loading real weights from {model_path}...")
    tensors = load_file(model_path)
    
    # Let's pick a layer. e.g. layer 0 MLP gate_proj
    # Layer 0 hidden_size=2560, intermediate_size=6912
    # gate_proj: [intermediate_size, hidden_size] -> [6912, 2560]
    # Packed shape in safetensors: [1728, 2560] (6912 / 4 = 1728)
    
    prefix = "model.layers.0.mlp.gate_proj"
    w_packed_hf = tensors[f"{prefix}.weight"]
    w_scale = tensors[f"{prefix}.weight_scale"].float().item()
    
    out_features = 6912
    in_features = 2560
    
    print(f"Layer: {prefix}, Out={out_features}, In={in_features}")
    
    # 1. Unpack for reference
    w_unpacked = unpack_weights_hf(w_packed_hf, out_features, in_features)
    
    # 2. Prepare Input (Int8 activations as per BitNet)
    M = 1
    x_pt = torch.randint(-128, 127, (M, in_features), dtype=torch.int8)
    x_pt_f = x_pt.float()
    
    # 3. Reference Result (PyTorch)
    # BitNet formula: Y = (X * W) * weight_scale / input_scale
    # In our engine, we handle (X * W) * weight_scale. Input scale is usually handled externally or merged.
    # Looking at modeling_bitnet.py: output = F.linear(input_quant, w_quant) * weight_scale
    
    print("Running PyTorch Reference...")
    start = time.time()
    ref_out = F.linear(x_pt_f, w_unpacked) * w_scale
    ref_time = time.time() - start
    
    # 4. VNN Execution
    print("Preparing VNN...")
    # Re-pack weights for VNN
    w_packed_vnn = pack_weights_vnn(w_unpacked)
    
    # Create VNN Tensors
    # Activation must be Int8
    x_vnn = vnn.Tensor(x_pt.numpy().astype(np.float32), dtype="int8", name="input")
    
    # Weight must be BitNet2 (uint8 packed)
    weight_vnn = vnn.Tensor(w_packed_vnn.astype(np.float32), dtype="bitnet2", shape=(out_features, in_features), name="weight")
    
    # Scale must be (N,)
    scale_vnn = vnn.Tensor(np.full((out_features,), w_scale, dtype=np.float32), name="scale")
    
    # Run on CPU first
    print("Running VNN CPU...")
    # Note: .bit_linear() is an instance method now
    start = time.time()
    vnn_out_cpu = x_vnn.bit_linear(weight_vnn, scale_vnn)
    vnn_time_cpu = time.time() - start
    
    vnn_out_cpu_np = vnn_out_cpu.numpy()
    
    # Run on GPU
    try:
        print("Running VNN GPU (Bonaire)...")
        x_vga = x_vnn.to("vga")
        w_vga = weight_vnn.to("vga")
        s_vga = scale_vnn.to("vga")
        
        start = time.time()
        vnn_out_gpu = x_vga.bit_linear(w_vga, s_vga)
        vnn_time_gpu = time.time() - start
        
        vnn_out_gpu_np = vnn_out_gpu.to("cpu").numpy()
    except Exception as e:
        print(f"GPU Error: {e}")
        vnn_out_gpu_np = None

    # 5. Compare
    ref_np = ref_out.numpy()
    
    def check(name, val):
        if val is None: return
        diff = np.abs(ref_np - val)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"--- {name} Results ---")
        print(f"Max Diff: {max_diff:.6f}")
        print(f"Mean Diff: {mean_diff:.6f}")
        if max_diff < 1e-4:
            print("PASS ✅")
        else:
            print("FAIL ❌")
            # Print a few samples
            print("Ref:", ref_np.flatten()[:5])
            print("VNN:", val.flatten()[:5])

    check("CPU", vnn_out_cpu_np)
    if vnn_out_gpu_np is not None:
        check("GPU", vnn_out_gpu_np)
        print(f"Speedup GPU vs CPU: {vnn_time_cpu / vnn_time_gpu:.2f}x")

if __name__ == "__main__":
    test_real_layer_parity()
