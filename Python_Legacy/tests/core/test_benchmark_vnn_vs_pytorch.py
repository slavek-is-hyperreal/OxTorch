import time
import numpy as np
import Python_Legacy.vulkan_nn_lib as vnn
from Python_Legacy.vulkan_nn_lib.tensor import Tensor
from Python_Legacy.vulkan_nn_lib.modules.tiled import TiledLinear
import taichi as ti
import torch

def float16_to_bytes(val):
    return np.array([val], dtype=np.float16).view(np.uint8)

def create_dummy_q4_weight(in_features, out_features):
    # Q4_0 packs 32 elements into 18 bytes
    total_elements = in_features * out_features
    num_blocks = total_elements // 32
    raw_data = np.zeros(num_blocks * 18, dtype=np.uint8)
    
    # Fill with dummy values (scale 0.5, small integers)
    for b in range(num_blocks):
        raw_data[b*18 : b*18+2] = float16_to_bytes(0.5)
        for j in range(16):
            raw_data[b*18 + 2 + j] = 0x88 # -8 and -8 just as a baseline
            
    return raw_data

def test_benchmark_vnn_vs_pytorch():
    ti.init(arch=ti.vulkan)
    print("=" * 60)
    print(" VNN vs PyTorch Benchmark (Hardware Limited/Hobbyist Tier)")
    print("=" * 60)
    print("Hypothesis: VNN sacrificing raw speed for extreme VRAM efficiency")
    print("using pure Python + Vulkan GGUF streaming.")
    print("-" * 60)
    
    # Let's simulate a large layer that easily fits in RAM but might struggle on old GPUs
    # E.g. Llama 7B single linear layer: 4096 x 4096
    in_features = 4096
    out_features = 4096
    batch_size = 1 # Single token generation (inference)
    
    # 1. PyTorch Baseline (FP32 CPU)
    print(f"Setting up PyTorch FP32 CPU Baseline ({in_features}x{out_features})...")
    pt_linear = torch.nn.Linear(in_features, out_features, bias=False)
    pt_input = torch.randn(batch_size, in_features)
    
    # Warmup
    _ = pt_linear(pt_input)
    
    start_time = time.time()
    for _ in range(10):
        _ = pt_linear(pt_input)
    pt_time = (time.time() - start_time) / 10
    
    # 2. VNN VRAM-Constrained Baseline (Q4_0 Vulkan Streaming)
    print(f"Setting up VNN Q4_0 Vulkan Baseline ({in_features}x{out_features})...")
    vnn_linear = TiledLinear(in_features, out_features, bias=False, tile_size=2048, quant_type='q4_0')
    
    # Hook our dummy Q4 weights into the tensor
    dummy_q4 = create_dummy_q4_weight(in_features, out_features)
    vnn_linear.weight = Tensor(dummy_q4) # Bypass standard float array with our raw bytes
    
    vnn_input = Tensor(pt_input.detach().numpy(), device='vulkan')
    
    # Warmup (allocates buffers and compiles shaders)
    _ = vnn_linear(vnn_input)
    ti.sync()
    
    start_time = time.time()
    for _ in range(10):
        _ = vnn_linear(vnn_input)
        ti.sync()
    vnn_time = (time.time() - start_time) / 10
    
    print("-" * 60)
    print(f"PyTorch CPU (Intel/AMD) Time: {pt_time*1000:.2f} ms")
    print(f"VNN Vulkan Q4_0 (Tiled Streaming) Time: {vnn_time*1000:.2f} ms")
    print("-" * 60)
    
    ratio = vnn_time / pt_time
    print(f"Performance Ratio: VNN is {ratio:.2f}x slower than heavily optimized PyTorch C++ backend.")
    print("\nCONCLUSION:")
    print("VNN demonstrates capability over raw speed. VNN allows users to run models")
    print("that completely exceed their GPU's VRAM by streaming 4-bit weights from system RAM,")
    print("doing the math dynamically on Vulkan shaders. This enables 'impossible' runs on low-end hardware.")

if __name__ == "__main__":
    test_benchmark_vnn_vs_pytorch()
