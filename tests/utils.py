import torch
import numpy as np
import vulkannn_rusted as vnn

def to_vnn(t: torch.Tensor, device="cpu"):
    """Convert a PyTorch tensor to a VNN Rusted Tensor."""
    return vnn.Tensor(t.detach().numpy(), device=device)

def check_close(v_t: vnn.Tensor, t_t: torch.Tensor, name="Tensor", atol=1e-5):
    """Verify parity between VNN Rusted Tensor and PyTorch/NumPy."""
    v_np = v_t.to_numpy()
    if hasattr(t_t, 'detach'): t_np = t_t.detach().numpy()
    else: t_np = t_t
    
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol)
        print(f"✓ {name} matches")
    except AssertionError as e:
        print(f"✗ {name} MISMATCH")
        # Provide more context on failure
        diff = np.abs(v_np - t_np)
        print(f"  Max Diff: {np.max(diff)}, Mean Diff: {np.mean(diff)}")
        raise e
