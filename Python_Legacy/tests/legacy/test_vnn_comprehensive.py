import os
import sys
import unittest
import numpy as np
import torch
import time
import Python_Legacy.vulkan_nn_lib as vnn
from Python_Legacy.vulkan_nn_lib.tensor import Tensor
from Python_Legacy.vulkan_nn_lib.memory import MemoryManager

class VNNComprehensiveTest(unittest.TestCase):
    """
    Comprehensive test suite for VNN spanning all devices, precisions and scales.
    Ensures mathematical parity with PyTorch.
    """

    @classmethod
    def setUpClass(cls):
        print("\n=== Initializing VNN Comprehensive Suite ===")
        Tensor.setup_ssd_storage()
        # Ensure we have enough VRAM for medium tests
        vram = MemoryManager.get_vram_budget()
        print(f"VRAM Budget: {vram/1e6:.1f}MB")

    def check_parity(self, v_res, t_expected, op_name, rtol=1e-4, atol=1e-4):
        """Standardized parity checker with scale-aware sampling for large tensors."""
        if v_res.total_size > 1e7: # > 10M elements (~40MB FP32)
             # OOM-Safe Sampling: Check first 1000, last 1000 and 1000 random elements
             print(f"  [Verify] Large tensor ({v_res.total_size/1e6:.1f}M elements). Using stratified sampling...")
             indices = np.concatenate([
                 np.arange(1000), 
                 np.arange(v_res.total_size - 1000, v_res.total_size),
                 np.random.randint(0, v_res.total_size, 1000)
             ])
             # Clamp indices
             indices = np.clip(indices, 0, v_res.total_size - 1)
             
             v_sample = v_res.get_samples(indices)
             t_sample = t_expected.detach().cpu().numpy().flatten()[indices]
             diff = np.abs(v_sample - t_sample).max()
        else:
             v_np = v_res.to_numpy()
             t_np = t_expected.detach().cpu().numpy()
             diff = np.abs(v_np - t_np).max()
        
        if diff >= atol + rtol * np.abs(t_sample if 't_sample' in locals() else t_np).max():
             print(f"  FAILED Parity: {op_name}")
             if 'v_sample' in locals():
                  print(f"    v_sample range: [{v_sample.min():.4f}, {v_sample.max():.4f}]")
                  print(f"    t_sample range: [{t_sample.min():.4f}, {t_sample.max():.4f}]")
             else:
                  print(f"    v_np range: [{v_np.min():.4f}, {v_np.max():.4f}]")
                  print(f"    t_np range: [{t_np.min():.4f}, {t_np.max():.4f}]")
             print(f"    Max Diff: {diff:.2e}")

        print(f"  [{op_name}] Max Diff: {diff:.2e}")
        self.assertTrue(diff < atol + rtol * np.abs(t_sample if 't_sample' in locals() else t_np).max(), 
                        f"Parity mismatch in {op_name}: {diff:.2e}")

    def run_op_matrix(self, op_func, op_name, sizes=['small'], devices=['cpu', 'vulkan', 'hybrid'], dtypes=[np.float32], is_binary=True, is_reduction=False):
        """Helper to run an op through a specific matrix of conditions."""
        for size_name in sizes:
            # Deterministic sizing
            if size_name == 'small':   n = 100 * 1024 # 0.4MB
            elif size_name == 'medium': n = 10 * 1024 * 1024 # 40MB
            elif size_name == 'large':  n = 200 * 1024 * 1024 # 800MB
            
            for dev in devices:
                for dtype in dtypes:
                    # Skip vulkan for int4 for now (usually handled by CPU unpack/IO)
                    if dev == 'vulkan' and dtype == 'int4': continue
                    
                    with self.subTest(size=size_name, device=dev, dtype=dtype):
                        print(f"\n[Test] {op_name} | {size_name} ({n/1e6:.1f}M) | {dev} | {dtype}")
                        
                        # Configuration overrides
                        os.environ["VNN_KAGGLE_MODE"] = "1" if dev == 'kaggle' else "0"
                        
                        # Data generation
                        if dtype == 'int4':
                             a_np = np.random.randint(-8, 7, n).astype(np.float32)
                             b_np = np.random.randint(-8, 7, n).astype(np.float32) if is_binary else None
                        else:
                             a_np = (np.random.randn(n) * 2).astype(dtype)
                             b_np = (np.random.randn(n) * 2).astype(dtype) if is_binary else None
                        
                        a_torch = torch.from_numpy(a_np).requires_grad_(True)
                        b_torch = torch.from_numpy(b_np).requires_grad_(True) if is_binary else None
                        
                        effective_dev = 'auto' if (size_name == 'large' or dev == 'hybrid') else dev
                        if effective_dev == 'auto':
                             MemoryManager._force_budget_bytes = int(n * 4 * 0.4)
                        else:
                             MemoryManager._force_budget_bytes = None

                        a_vnn = Tensor(a_np, device=effective_dev, dtype=dtype, requires_grad=True)
                        b_vnn = Tensor(b_np, device=effective_dev, dtype=dtype, requires_grad=True) if is_binary else None
                        
                        # 1. Forward Pass
                        start = time.perf_counter()
                        res_vnn = op_func(a_vnn, b_vnn) if is_binary else op_func(a_vnn)
                        elapsed = time.perf_counter() - start
                        
                        res_torch = op_func(a_torch, b_torch) if is_binary else op_func(a_torch)
                        
                        # Apply quantization to ground truth for INT4 parity
                        if dtype == 'int4':
                             res_torch = (res_torch + 8.0).clamp(0, 15).floor() - 8.0
                             
                        # Tolerance adjustment
                        cur_rtol, cur_atol = 1e-4, 1e-4
                        if dtype == np.float16: cur_rtol, cur_atol = 5e-3, 5e-3
                        if dtype == 'int4': cur_rtol, cur_atol = 1e-2, 1e-2

                        print(f"    Forward: {elapsed:.4f}s")
                        self.check_parity(res_vnn, res_torch, f"{op_name}_Fwd", rtol=cur_rtol, atol=cur_atol)
                        
                        # 2. Backward Pass (if supported)
                        if not is_reduction and dtype != 'int4': # int4 doesn't support autograd well (discrete)
                             # Use random loss grad
                             grad_vnn_np = (np.random.randn(*res_vnn.shape) * 0.1).astype(np.float32)
                             grad_vnn = Tensor(grad_vnn_np)
                             grad_torch = torch.from_numpy(grad_vnn_np)
                             
                             t0 = time.perf_counter()
                             res_vnn.backward(grad_vnn)
                             print(f"    Backward: {time.perf_counter()-t0:.4f}s")
                             
                             res_torch.backward(grad_torch)
                             
                             # Check weight grads
                             self.check_parity(a_vnn.grad, a_torch.grad, f"{op_name}_GradA", rtol=cur_rtol, atol=cur_atol)
                             if is_binary:
                                  self.check_parity(b_vnn.grad, b_torch.grad, f"{op_name}_GradB", rtol=cur_rtol, atol=cur_atol)

                        # Cleanup
                        MemoryManager._force_budget_bytes = None

    def test_elementwise_add(self):
        self.run_op_matrix(lambda x, y: x + y, "Add", dtypes=[np.float32, np.float16, 'int4'])

    def test_elementwise_div(self):
        # Avoid zeros
        self.run_op_matrix(lambda x, y: x / (y.relu() + 0.1), "Div")

    def test_activations(self):
        import torch.nn.functional as F_t
        self.run_op_matrix(lambda x: x.relu(), "ReLU", is_binary=False)
        self.run_op_matrix(lambda x: x.silu() if hasattr(x, 'silu') else F_t.silu(x), "SiLU", is_binary=False)

    def test_matmul_comprehensive(self):
        """Special cases for MatMul tiling."""
        for dev in ['cpu', 'vulkan', 'hybrid']:
             # (M, K, N)
             configs = [(64, 128, 64), (1024, 1024, 1024)] # Small, Large (triggers SSD/Tiling)
             for M, K_DIM, N in configs:
                  with self.subTest(dev=dev, M=M, K=K_DIM, N=N):
                       a_np = np.random.randn(M, K_DIM).astype(np.float32)
                       b_np = np.random.randn(K_DIM, N).astype(np.float32)
                       
                       # Force low budget for MatMul tiling verification
                       if M > 500: MemoryManager._force_budget_bytes = 10 * 1024 * 1024 # 10MB
                       
                       a_v = Tensor(a_np, device=dev if dev != 'hybrid' else 'auto')
                       b_v = Tensor(b_np, device=dev if dev != 'hybrid' else 'auto')
                       
                       res_v = a_v @ b_v
                       res_t = torch.from_numpy(a_np) @ torch.from_numpy(b_np)
                       
                       self.check_parity(res_v, res_t, f"MatMul_{M}x{K_DIM}x{N}_{dev}")
                       MemoryManager._force_budget_bytes = None

    def test_reductions(self):
        # Full reduction
        self.run_op_matrix(lambda x, y=None: x.sum(), "Sum", is_binary=False, is_reduction=True)
        self.run_op_matrix(lambda x, y=None: x.mean(), "Mean", is_binary=False, is_reduction=True)
        
        # Last dimension reduction
        def mean_last(x): return x.mean(dim=-1)
        # Test with 2D shape for last dim mean
        n = 1024 * 128
        a_np = np.random.randn(128, 1024).astype(np.float32)
        a_v = Tensor(a_np, device='vulkan')
        self.check_parity(a_v.mean(dim=-1), torch.from_numpy(a_np).mean(dim=-1), "Mean_Dim_Vulkan")

    def test_kaggle_offload(self):
        """Dedicated Kaggle test if enabled."""
        if os.getenv("KAGGLE_TEST") != "1":
             self.skipTest("KAGGLE_TEST not set to 1")
        
        # Test one large op on Kaggle
        self.run_op_matrix(lambda x, y: x + y, "Add_Kaggle", sizes=['large'], devices=['kaggle'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['fast', 'full', 'kaggle'], default='fast')
    args, unknown = parser.parse_known_args()
    
    # Global config for test methods
    test_sizes = ['small']
    if args.mode == 'full':
         test_sizes = ['small', 'medium', 'large']
    
    class ConfiguredVNNTest(VNNComprehensiveTest):
        def test_elementwise_add(self):
            self.run_op_matrix(lambda x, y: x + y, "Add", sizes=test_sizes, dtypes=[np.float32, np.float16, 'int4'])

        def test_elementwise_div(self):
            self.run_op_matrix(lambda x, y: x / (y.relu() + 0.1), "Div", sizes=test_sizes)

        def test_activations(self):
            import torch.nn.functional as F_t
            self.run_op_matrix(lambda x: x.relu(), "ReLU", sizes=test_sizes, is_binary=False)
            self.run_op_matrix(lambda x: x.silu() if hasattr(x, 'silu') else F_t.silu(x), "SiLU", sizes=test_sizes, is_binary=False)

        def test_reductions(self):
            self.run_op_matrix(lambda x, y=None: x.sum(), "Sum", sizes=test_sizes, is_binary=False, is_reduction=True)
            self.run_op_matrix(lambda x, y=None: x.mean(), "Mean", sizes=test_sizes, is_binary=False, is_reduction=True)

    # Configure Kaggle env
    if args.mode == 'kaggle':
         os.environ["KAGGLE_TEST"] = "1"
    else:
         os.environ["KAGGLE_TEST"] = "0"

    sys.argv = [sys.argv[0]] + unknown
    unittest.main(defaultTest='ConfiguredVNNTest' if args.mode != 'kaggle' else None)
