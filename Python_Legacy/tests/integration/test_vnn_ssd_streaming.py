import torch
import numpy as np
import Python_Legacy.vulkan_nn_lib.torch_shim as vnn
from Python_Legacy.vulkan_nn_lib.memory import MemoryManager
from Python_Legacy.vulkan_nn_lib.tensor import Tensor
import time
import os

# --- MOCKING MEMORY LIMITS ---
# We force tiling and SSD offloading for small tensors
def mock_get_safe_budget():
    return 64 * 1024 * 1024 # 64MB budget

def mock_should_tile(size_bytes):
    return size_bytes > 8 * 1024 * 1024 # Tile if > 8MB

# Apply mocks
MemoryManager.get_safe_budget = mock_get_safe_budget
MemoryManager.should_tile = mock_should_tile

def check_parity(name, v_t, t_t, atol=1e-5):
    v_np = v_t.to_numpy()
    if isinstance(t_t, torch.Tensor):
        t_np = t_t.detach().numpy()
    else:
        t_np = t_t
    
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, err_msg=f"Parity failure in {name}")
        print(f"  ✓ {name}: Passed")
        return True, ""
    except AssertionError as e:
        print(f"  ✗ {name}: FAILED")
        return False, str(e)

def run_ssd_suite():
    print("="*60)
    print("Starting Phase 5: SSD Streaming & High Pressure Parity Tests")
    print(f"MOCKED BUDGET: {MemoryManager.get_safe_budget()/1e6:.1f}MB")
    print("="*60)

    results = []

    # 1. SSD Offloading Test (Explicit device='ssd')
    print("\n[SCENARIO 1] Explicit device='ssd'")
    # 128MB tensor (16M float32)
    N = 16 * 1024 * 1024 
    a_np = np.random.randn(N).astype(np.float32)
    b_np = np.random.randn(N).astype(np.float32)
    
    vt_a = vnn.tensor(a_np, device='ssd', requires_grad=True)
    vt_b = vnn.tensor(b_np, device='ssd', requires_grad=True)
    
    tt_a = torch.from_numpy(a_np).requires_grad_(True)
    tt_b = torch.from_numpy(b_np).requires_grad_(True)
    
    # Check ADD (triggers SOE elementwise)
    v_res = vt_a + vt_b
    t_res = tt_a + tt_b
    ok, err = check_parity("SSD_ADD_128MB", v_res, t_res)
    results.append(("SSD_ADD", ok, err))
    
    # Check SUM (triggers SOE reduction)
    v_sum = vt_a.sum()
    t_sum = tt_a.sum()
    print(f"  VNN Sum: {v_sum.to_numpy().item():.6f} | Torch Sum: {t_sum.item():.6f}")
    ok, err = check_parity("SSD_SUM_128MB", v_sum, t_sum, atol=5e-3)
    results.append(("SSD_SUM", ok, err))

    # 2. AUTO-Offloading Test (Triggered by mocked budget)
    print("\n[SCENARIO 2] Auto-Offloading (Large than mocked budget & Vulkan threshold)")
    # 200MB tensor > 128MB Vulkan threshold
    N2 = 50 * 1024 * 1024
    a_np2 = np.random.randn(N2).astype(np.float32)
    
    vt_auto = vnn.tensor(a_np2, device='auto')
    print(f"  Detected Device: {vt_auto.device} (Expected: ssd)")
    ok = (vt_auto.device == 'ssd')
    results.append(("AUTO_OFFLOAD", ok, "" if ok else f"Got {vt_auto.device}"))
    
    # 3. Autograd on SSD
    print("\n[SCENARIO 3] Autograd on SSD-Resident Tensors")
    v_sum.backward() # v_sum is scalar
    t_sum.backward()
    
    # Grad check
    print(f"  VNN Grad Mean: {vt_a.grad.to_numpy().mean():.6f} | Expected: 1.0")
    ok, err = check_parity("SSD_GRAD_A", vt_a.grad, tt_a.grad)
    results.append(("SSD_GRAD", ok, err))

    # 4. LARGE MatMul (Tiled)
    print("\n[SCENARIO 4] Tiled MatMul (Hybrid/Streaming)")
    # Shape that doesn't fit in cached Vulkan ndarray but fits in RAM or triggers tiling
    # 4096 x 4096 = 16M elements (64MB)
    size = 4096
    m_a_np = np.random.randn(size, size).astype(np.float32)
    m_b_np = np.random.randn(size, size).astype(np.float32)
    
    vt_m_a = vnn.tensor(m_a_np, device='ssd')
    vt_m_b = vnn.tensor(m_b_np, device='ssd')
    tt_m_a = torch.from_numpy(m_a_np)
    tt_m_b = torch.from_numpy(m_b_np)
    
    print(f"  Running 4096x4096 MatMul (SSD -> Tiled CPU)...")
    t0 = time.perf_counter()
    v_m_res = vt_m_a @ vt_m_b
    dur = time.perf_counter() - t0
    print(f"    Done in {dur:.2f}s")
    
    t_m_res = tt_m_a @ tt_m_b
    ok, err = check_parity("SSD_MATMUL_4096", v_m_res, t_m_res, atol=1e-2) # Higher atol for MatMul
    results.append(("SSD_MATMUL", ok, err))

    # Summary
    print("\n" + "="*60)
    print("Phase 5 Summary")
    print("="*60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed < total:
        print("\nFailures:")
        for name, ok, err in results:
            if not ok:
                print(f"- {name}: {err[:150]}...")

if __name__ == "__main__":
    run_ssd_suite()
