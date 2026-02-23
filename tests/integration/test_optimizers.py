import torch
import numpy as np
import vulkan_nn_lib.core as vnn
from tests.utils import check_close, to_vnn, check_grads

def test_sgd():
    print("\n--- Testing SGD (Vulkan vs Torch) ---")
    W_shape = (10, 10)
    lr = 0.1
    
    # Common initialization
    init_wt = torch.randn(W_shape)
    gt = torch.randn(W_shape)
    
    # torch
    wt = init_wt.clone().requires_grad_(True)
    wt.grad = gt.clone()
    
    opt = torch.optim.SGD([wt], lr=lr)
    opt.step()
    
    # vnn
    wv = to_vnn(init_wt, requires_grad=True)
    gv = to_vnn(gt)
    wv.grad = gv
    
    print(f"Pre-step VV: {wv.to_numpy()[0,:5]}")
    vopt = vnn.SGD([wv], lr=lr)
    vopt.step()
    print(f"Post-step VV: {wv.to_numpy()[0,:5]}")
    print(f"Post-step PT: {wt.detach().numpy()[0,:5]}")
    
    # Relax rtol for fp32 comparisons across backends
    check_close(wv, wt, "SGD Weight Update", atol=1e-4)

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
