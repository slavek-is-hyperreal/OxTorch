import torch
import numpy as np
import vulkan_nn_lib.core as vnn

def to_vnn(t: torch.Tensor, requires_grad=False):
    return vnn.Tensor(t.detach().numpy(), requires_grad=requires_grad)

def check_close(v_t: vnn.Tensor, t_t: torch.Tensor, name="Tensor", atol=1e-5):
    v_np = v_t.to_numpy()
    t_np = t_t.detach().numpy()
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol)
        print(f"✓ {name} matches")
    except AssertionError as e:
        print(f"✗ {name} MISMATCH")
        raise e

def check_grads(v_t: vnn.Tensor, t_t_or_grad, name="Gradient", atol=1e-5):
    # If t_t_or_grad is a torch tensor, try to get its .grad if it's a leaf.
    # Otherwise, assume it's the gradient itself.
    t_grad = t_t_or_grad
    if isinstance(t_t_or_grad, torch.Tensor):
        if t_t_or_grad.is_leaf and hasattr(t_t_or_grad, 'grad') and t_t_or_grad.grad is not None:
            t_grad = t_t_or_grad.grad
    
    if v_t.grad is None or t_grad is None:
        if v_t.grad is None and t_grad is None:
            print(f"✓ {name} (Both None)")
            return
        print(f"✗ {name} MISMATCH (One is None)")
        v_grad_val = "None" if v_t.grad is None else "Tensor"
        t_grad_val = "None" if t_grad is None else "Tensor"
        raise AssertionError(f"{name} mismatch: Vulkan grad is {v_grad_val}, Torch grad is {t_grad_val}")
    
    check_close(v_t.grad, t_grad, name=name, atol=atol)
