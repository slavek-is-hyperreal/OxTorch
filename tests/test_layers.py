import torch
import vulkan_nn_lib.core as vnn
from tests.utils import to_vnn, check_close, check_grads

def test_linear_backward():
    print("\n--- Testing Linear Backward ---")
    in_features, out_features = 4, 2
    batch_size = 3
    
    x = torch.randn(batch_size, in_features, requires_grad=True)
    layer = torch.nn.Linear(in_features, out_features)
    y = layer(x)
    y.sum().backward()
    
    vx = to_vnn(x, requires_grad=True)
    vlayer = vnn.Linear(in_features, out_features)
    # Load weights from torch to match
    vlayer.weight.arr.from_numpy(layer.weight.detach().numpy().T.flatten())
    vlayer.bias.arr.from_numpy(layer.bias.detach().numpy().flatten())
    vnn.ti.sync()
    
    vy = vlayer(vx)
    vy.backward(vnn.Tensor(np.ones((batch_size, out_features)), shape=(batch_size, out_features)))
    
    check_close(vy, y, "Linear Output")
    check_grads(vx, x, "Grad X")
    check_grads(vlayer.weight, layer.weight.grad.T, "Grad Weight")
    check_grads(vlayer.bias, layer.bias, "Grad Bias")

def test_rmsnorm_backward():
    print("\n--- Testing RMSNorm Backward ---")
    dim = 4
    batch_size = 2
    x = torch.randn(batch_size, dim, requires_grad=True)
    
    # Simple RMSNorm implementation in Torch for comparison
    def torch_rmsnorm(x, w, eps=1e-6):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x / rms) * (1.0 + w) # Gemma style add_unit_offset=True
    
    w = torch.randn(dim, requires_grad=True)
    y = torch_rmsnorm(x, w)
    y.sum().backward()
    
    vx = to_vnn(x, requires_grad=True)
    vw = to_vnn(w, requires_grad=True)
    vlayer = vnn.RMSNorm(dim)
    vlayer.weight = vw
    
    vy = vlayer(vx)
    vy.backward(vnn.Tensor(np.ones((batch_size, dim)), shape=(batch_size, dim)))
    
    check_close(vy, y, "RMSNorm Output")
    check_grads(vx, x, "Grad X")
    check_grads(vw, w, "Grad W")

import numpy as np
if __name__ == "__main__":
    test_linear_backward()
    test_rmsnorm_backward()
