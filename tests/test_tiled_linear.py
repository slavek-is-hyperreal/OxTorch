import torch
import numpy as np
import vulkan_nn_lib.core as vnn
from tests.utils import to_vnn, check_close, check_grads

def test_tiled_linear_vs_linear():
    print("\n--- Testing TiledLinear vs Linear ---")
    in_features, out_features = 128, 64
    batch_size = 4
    tile_size = 32
    
    # 1. Setup Torch
    x = torch.randn(batch_size, in_features, requires_grad=True)
    layer = torch.nn.Linear(in_features, out_features)
    y = layer(x)
    y.sum().backward()
    
    # 2. Setup standard Vulkan Linear
    vx = to_vnn(x, requires_grad=True)
    vlinear = vnn.Linear(in_features, out_features)
    vlinear.weight.load_from_numpy(layer.weight.detach().numpy().T)
    vlinear.bias.load_from_numpy(layer.bias.detach().numpy())
    
    vy = vlinear(vx)
    vy.backward(vnn.Tensor(np.ones((batch_size, out_features)), shape=(batch_size, out_features)))
    
    # 3. Setup TiledLinear
    vx_tiled = to_vnn(x, requires_grad=True)
    vtiled = vnn.TiledLinear(in_features, out_features, tile_size=tile_size)
    # Copy weights to TiledLinear weight_ram (which is now vtiled.weight.arr if device='cpu')
    vtiled.weight.arr[:] = layer.weight.detach().numpy().T.flatten()
    vtiled.bias.load_from_numpy(layer.bias.detach().numpy())
    
    vy_tiled = vtiled(vx_tiled)
    vy_tiled.backward(vnn.Tensor(np.ones((batch_size, out_features)), shape=(batch_size, out_features)))
    
    # 4. Compare
    print("Comparing TiledLinear vs Torch...")
    check_close(vy_tiled, y, "Output")
    check_grads(vx_tiled, x, "Grad X")
    check_grads(vtiled.weight, layer.weight.grad.T, "Grad Weight")
    check_grads(vtiled.bias, layer.bias, "Grad Bias")
    
    print("\nComparing TiledLinear vs Standard Linear (Vulkan)...")
    check_close(vy_tiled, vy.to_numpy(), "Tiled vs Standard Output")
    # check_grads(vx_tiled, vx, "Tiled vs Standard Grad X") # check_grads expects torch tensor as 2nd arg
    np.testing.assert_allclose(vx_tiled.grad.to_numpy(), vx.grad.to_numpy(), atol=1e-5)
    print("✓ Grad X (VNN vs VNN) matches")
    np.testing.assert_allclose(vtiled.weight.grad.to_numpy(), vlinear.weight.grad.to_numpy(), atol=1e-5)
    print("✓ Grad Weight (VNN vs VNN) matches")

if __name__ == "__main__":
    test_tiled_linear_vs_linear()
