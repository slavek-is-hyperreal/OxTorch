import Python_Legacy.vulkan_nn_lib.torch_shim as torch
import numpy as np
import time
import os

def test_auto_sgd_ssd():
    print("\n--- Testing AutoSGD SSD-Native (10MB) ---")
    N = 2_500_000 # 10MB
    lr = 0.1
    
    print(f"[1/4] Initializing 10MB parameter on SSD...")
    w = torch.randn(N, device='ssd', requires_grad=True)
    # Force some values to verify update
    w.arr[0:10] = 1.0
    
    print(f"[2/4] Forward/Backward to get grads...")
    loss = (w * 2.0).sum()
    loss.backward()
    
    print(f"[3/4] AutoSGD Step...")
    # Force AutoSGD to use hybrid/SSD strategy by setting budgets low
    from Python_Legacy.vulkan_nn_lib.optimizers import AutoSGD
    opt = AutoSGD([w], lr=lr, vram_budget=2*1024*1024, ram_budget=2*1024*1024)
    
    t0 = time.perf_counter()
    opt.step()
    print(f"      Step done in {time.perf_counter()-t0:.2f}s")
    
    print(f"[4/4] Verifying Weights...")
    # Expected: w = w - lr * grad = 1.0 - 0.1 * 2.0 = 0.8
    w_slice = w[0:10].to_numpy()
    print(f"      Weight slice: {w_slice}")
    
    if np.allclose(w_slice, 0.8, atol=1e-5):
        print("  [PASS] AutoSGD SSD Math Correct!")
    else:
        print("  [FAIL] Math mismatch!")

def test_auto_adam_ssd():
    print("\n--- Testing AutoAdam SSD-Native (10MB) ---")
    N = 2_500_000 # 10MB
    lr = 1e-3
    
    print(f"[1/4] Initializing 10MB parameter on SSD...")
    w = torch.randn(N, device='ssd', requires_grad=True)
    w.arr[0:10] = 1.0
    
    print(f"[2/4] Forward/Backward to get grads...")
    loss = (w * 1.0).sum()
    loss.backward()
    
    print(f"[3/4] AutoAdam Step...")
    from Python_Legacy.vulkan_nn_lib.optimizers import AutoAdam
    opt = AutoAdam([w], lr=lr, vram_budget=2*1024*1024, ram_budget=4*1024*1024)
    
    # Run 1 step
    t0 = time.perf_counter()
    opt.step()
    print(f"      Step done in {time.perf_counter()-t0:.2f}s")
    
    print(f"[4/4] Verifying Weights...")
    # For 1st step with grad=1, m = (1-b1)*1 = 0.1, v = (1-b2)*1*1 = 0.001
    # m_hat = 0.1 / (1-0.9) = 1.0, v_hat = 0.001 / (1-0.999) = 1.0
    # update = lr * 1.0 / (sqrt(1.0) + eps) approx lr
    # new_w approx 1.0 - 0.001 = 0.999
    w_slice = w[0:10].to_numpy()
    print(f"      Weight slice: {w_slice}")
    
    if np.allclose(w_slice, 1.0 - lr, atol=1e-4):
        print("  [PASS] AutoAdam SSD Math Correct!")
    else:
        print("  [FAIL] Math mismatch!")

if __name__ == "__main__":
    test_auto_sgd_ssd()
    test_auto_adam_ssd()
