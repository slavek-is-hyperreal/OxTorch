import torch
import torch.nn.functional as F_torch
import numpy as np
import Python_Legacy.vulkan_nn_lib.core as vnn
from tests.utils import to_vnn, check_close, check_grads

def test_softmax_backward():
    print("\n--- Testing Softmax Backward ---")
    shape = (2, 5)
    x = torch.randn(shape, requires_grad=True)
    y = torch.softmax(x, dim=-1)
    y.sum().backward()
    
    vx = to_vnn(x, requires_grad=True)
    vlayer = vnn.Softmax(dim=-1)
    vy = vlayer(vx)
    # vnn.Softmax implementation returns a new tensor, so we can call backward on it.
    # But wait, backward() on non-scalar requires gradient argument.
    # torch .sum().backward() is equivalent to backward(torch.ones_like(y)) implicitly for scalar output of sum.
    # vnn does not have sum() autograd support in this test script yet?
    # vnn.Tensor.backward() supports explicit gradient.
    grad_output = vnn.Tensor(np.ones(shape), shape=shape)
    vy.backward(grad_output)
    
    check_close(vy, y, "Softmax Output")
    check_grads(vx, x, "Grad X")

def test_leaky_relu_backward():
    print("\n--- Testing LeakyReLU Backward ---")
    shape = (10,)
    x = torch.randn(shape, requires_grad=True)
    # Ensure some positive and negative values to test both branches
    x.data[0] = 1.0
    x.data[1] = -1.0
    
    alpha = 0.1
    y = F_torch.leaky_relu(x, negative_slope=alpha)
    y.sum().backward()
    
    vx = to_vnn(x, requires_grad=True)
    vlayer = vnn.LeakyReLU(negative_slope=alpha)
    vy = vlayer(vx)
    vy.backward(vnn.Tensor(np.ones(shape), shape=shape))
    
    check_close(vy, y, "LeakyReLU Output")
    check_grads(vx, x, "Grad X")

def test_gelu_tanh_backward():
    print("\n--- Testing GELUTanh Backward ---")
    shape = (10,)
    x = torch.randn(shape, requires_grad=True)
    
    # PyTorch GELU 'tanh' approximation
    y = F_torch.gelu(x, approximate='tanh')
    y.sum().backward()
    
    vx = to_vnn(x, requires_grad=True)
    vlayer = vnn.GELUTanh() 
    vy = vlayer(vx)
    vy.backward(vnn.Tensor(np.ones(shape), shape=shape))
    
    check_close(vy, y, "GELUTanh Output", atol=1e-3)
    check_grads(vx, x, "Grad X", atol=1e-3)

if __name__ == "__main__":
    test_softmax_backward()
    test_leaky_relu_backward()
    test_gelu_tanh_backward()
