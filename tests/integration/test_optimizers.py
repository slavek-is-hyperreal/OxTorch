import torch
import numpy as np
import vulkan_nn_lib.core as vnn
from tests.utils import check_close, to_vnn, check_grads

def test_sgd():
    print("\n--- Testing SGD (Vulkan vs Torch) ---")
    W_shape = (10, 10)
    lr = 0.1
    
    # torch
    wt = torch.randn(W_shape, requires_grad=True)
    gt = torch.randn(W_shape)
    wt.grad = gt.clone()
    
    opt = torch.optim.SGD([wt], lr=lr)
    opt.step()
    
    # vnn
    wv = to_vnn(wt + lr * gt, requires_grad=True) # Start from same point
    gv = to_vnn(gt)
    wv.grad = gv
    
    vopt = vnn.SGD([wv], lr=lr)
    vopt.step()
    
    check_close(wv, wt, "SGD Weight Update")

def test_adam_vram():
    print("\n--- Testing Adam (VRAM Resident vs Torch) ---")
    W_shape = (10, 10)
    lr = 1e-3
    
    # torch
    wt = torch.randn(W_shape, requires_grad=True)
    opt = torch.optim.Adam([wt], lr=lr)
    
    # vnn
    wv = to_vnn(wt, requires_grad=True)
    vopt = vnn.Adam([wv], lr=lr)
    
    for i in range(3):
        print(f"Step {i+1}")
        gt = torch.randn(W_shape)
        wt.grad = gt.clone()
        wv.grad = to_vnn(gt)
        
        opt.step()
        vopt.step()
        
        check_close(wv, wt, f"Adam Update (VRAM) Step {i+1}", atol=1e-4)

def test_adam_ram():
    print("\n--- Testing Adam (RAM Paged vs Torch) ---")
    W_shape = (10, 10)
    lr = 1e-3
    
    # torch
    wt = torch.randn(W_shape, requires_grad=True)
    opt = torch.optim.Adam([wt], lr=lr)
    
    # vnn: Force device='ram'
    wv = vnn.Tensor(wt.detach().numpy(), device='ram', requires_grad=True)
    vopt = vnn.Adam([wv], lr=lr)
    
    for i in range(3):
        print(f"Step {i+1}")
        gt = torch.randn(W_shape)
        wt.grad = gt.clone()
        wv.grad = vnn.Tensor(gt.numpy(), device='ram')
        
        opt.step()
        vopt.step()
        
        check_close(wv, wt, f"Adam Update (RAM) Step {i+1}", atol=1e-4)

if __name__ == "__main__":
    vnn.warmup()
    test_sgd()
    test_adam_vram()
    test_adam_ram()
