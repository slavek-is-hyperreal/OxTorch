import vulkan_torch as torch
import torch.nn as nn
import numpy as np
import time

def benchmark_paging():
    print("--- VulkanNN: Virtual VRAM (Weight Paging) Benchmark ---")
    
    # Dimensions: 2048x4096 layer
    # This matrix alone takes ~32MB. In a real LLM, you'd have many such layers.
    # Tile size 1024 means we'll stream it in 4 chunks of 1024 columns.
    IN_DIM = 2048
    OUT_DIM = 4096
    TILE_SIZE = 1024
    
    # 1. Standard Linear (Full VRAM)
    print(f"Initializing Standard Linear ({IN_DIM}x{OUT_DIM})...")
    linear = nn.Linear(IN_DIM, OUT_DIM)
    
    # 2. Tiled Linear (Paging from RAM)
    print(f"Initializing Tiled Linear ({IN_DIM}x{OUT_DIM}, Tile={TILE_SIZE})...")
    tiled_linear = nn.TiledLinear(IN_DIM, OUT_DIM, tile_size=TILE_SIZE)
    
    # Sync weights for comparison
    weights = linear.weight.to_numpy()
    tiled_linear.weight_ram = weights.copy()
    if linear.has_bias:
        tiled_linear.bias.arr.from_numpy(linear.bias.to_numpy())
    
    # Input
    x_np = np.random.randn(32, IN_DIM).astype(np.float32)
    x = torch.from_numpy(x_np)
    
    # Inference: Standard
    start = time.time()
    out1 = linear(x)
    t1 = time.time() - start
    print(f"Standard Linear Time: {t1:.4f}s")
    
    # Inference: Tiled (Virtual VRAM)
    start = time.time()
    out2 = tiled_linear(x)
    t2 = time.time() - start
    print(f"Tiled Linear (Paging) Time: {t2:.4f}s")
    
    # Validation
    diff = np.abs(out1.to_numpy() - out2.to_numpy()).max()
    print(f"Maximum absolute difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("SUCCESS: Virtual VRAM math matches standard VRAM math!")
    else:
        print("FAILURE: Math mismatch detected.")

if __name__ == "__main__":
    benchmark_paging()
