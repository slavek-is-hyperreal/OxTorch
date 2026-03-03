import Python_Legacy.vulkan_nn_lib.torch_shim as torch
import numpy as np
import time
import os

def test_ssd_autograd():
    print("\n--- Phase 4: SSD-Native Autograd Verification ---")
    
    # 1. Initialize a large SSD tensor (1GB)
    # 250M elements * 4 bytes = 1GB
    N = 250_000_000
    print(f"[1/4] Initializing 1GB parameter on SSD...")
    w = torch.randn(N, device='ssd', requires_grad=True)
    
    # 2. Perform a simple operation that produces a loss
    print(f"[2/4] Forward pass (SSD -> Scalar)...")
    t0 = time.perf_counter()
    loss = (w * 2.0).sum()
    print(f"      Forward done in {time.perf_counter()-t0:.2f}s | Loss: {loss.item():.2f}")
    
    # 3. Backward pass
    print(f"[3/4] Backward pass (SSD-Native Accumulation)...")
    t0 = time.perf_counter()
    loss.backward()
    print(f"      Backward done in {time.perf_counter()-t0:.2f}s")
    
    # 4. Verify Gradient
    print(f"[4/4] Verifying gradient on SSD...")
    if w.grad is None:
        print("  [FAIL] w.grad is None!")
        return
    
    print(f"      Grad Device: {w.grad.device}")
    grad_slice = w.grad[0:10].to_numpy()
    print(f"      Grad slice (expected 2.0): {grad_slice}")
    
    if np.allclose(grad_slice, 2.0):
        print("  [PASS] SSD Autograd Math Correct!")
    else:
        print("  [FAIL] Math mismatch!")

if __name__ == "__main__":
    test_ssd_autograd()
