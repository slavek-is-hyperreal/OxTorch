"""
tests/test_sprint1_ops.py — Parity tests for Sprint 1 MLP ops.

Covers every new operation added in Sprint 1:
  - Group 0: ReLU (AVX1 path verification)
  - Group 1: mul / sub / div / scalar ops
  - Group 2: reshape / view / squeeze / unsqueeze / flatten
  - Group 3: gelu / leaky_relu / elu / tanh / clamp
  - Group 4: sum / mean / max / min
  - Group 5: softmax / log_softmax
  - Group 6: Tensor creators (zeros / ones / full / rand / randn)

Rule: every test assert np.allclose(..., atol=1e-4) for F32,
      atol=1e-2 for F16/BF16 (reduced precision expected).

Run with:
    cd /my_data/gaussian_room
    PYTHONPATH=. pytest tests/test_sprint1_ops.py -v
"""

import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Import VNN — tries main branch first, falls back to exp
try:
    import vulkannn_rusted_main as vnn
    from vulkannn_rusted_main import Tensor, DataType
except ImportError:
    import vulkannn_rusted_exp as vnn
    from vulkannn_rusted_exp import Tensor, DataType

DEVICES = ["cpu", "vulkan"]   # "hybrid" added where applicable
F32 = DataType.F32
F16 = DataType.F16
BF16 = DataType.BF16

DTYPES = [
    (F32, np.float32, torch.float32, 1e-4),
    (F16, np.float16, torch.float16, 1e-2),
    (BF16, np.float32, torch.bfloat16, 1e-1), # Numpy lacks bf16, use float32 locally
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vnn(arr: np.ndarray, dtype=F32, device="cpu") -> Tensor:
    return Tensor(arr.astype(np.float32), dtype=dtype, device=device)

def make_torch(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)

def atol_for(dtype) -> float:
    return 1e-2 if dtype in (F16, BF16) else 1e-4


# ===========================================================================
# GROUP 0: ReLU (verify AVX1 vmaxps path still gives correct results)
# ===========================================================================

class TestReLU:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", [F32])
    def test_relu_basic(self, device, dtype):
        """Basic parity: negatives → 0, positives unchanged."""
        data = np.array([-3.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
        t = make_vnn(data, dtype, device)
        result = t.relu().to_numpy().flatten()
        expected = np.maximum(data, 0.0)
        assert np.allclose(result, expected, atol=atol_for(dtype)), \
            f"ReLU mismatch ({device}, {dtype}): {result} vs {expected}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_relu_large(self, device):
        """1M elements — exercises AVX1 path (and Vulkan PCIe path for vulkan device)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1_000_000).astype(np.float32)
        t = make_vnn(data, F32, device)
        result = t.relu().to_numpy().flatten()
        expected = np.maximum(data, 0.0)
        assert np.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_relu_all_negative(self, device):
        data = np.full(256, -5.0, dtype=np.float32)
        result = make_vnn(data, F32, device).relu().to_numpy().flatten()
        assert np.all(result == 0.0), f"All-negative ReLU should be 0: {result[:5]}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_relu_all_positive(self, device):
        data = np.full(256, 3.0, dtype=np.float32)
        result = make_vnn(data, F32, device).relu().to_numpy().flatten()
        assert np.allclose(result, 3.0)

    def test_relu_inplace_parity(self):
        """relu_into must match relu output."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal(10_000).astype(np.float32)
        t_src  = make_vnn(data, F32, "cpu")
        t_out  = make_vnn(np.zeros_like(data), F32, "cpu")
        t_src.relu_into(t_out)
        expected = make_vnn(data, F32, "cpu").relu().to_numpy()
        assert np.allclose(t_out.to_numpy(), expected, atol=1e-5)

    @pytest.mark.parametrize("dtype", [F16, BF16])
    def test_relu_half_precision(self, dtype):
        """F16/BF16 ReLU parity."""
        data = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        result = make_vnn(data, dtype, "cpu").relu().to_numpy().flatten()
        expected = np.maximum(data, 0.0)
        assert np.allclose(result, expected, atol=atol_for(dtype))


# ===========================================================================
# GROUP 1: Elementwise mul / sub / div / scalar
# (Tests will be enabled once the ops are implemented — marked xfail until then)
# ===========================================================================

class TestElementwise:
    @pytest.mark.parametrize("device", DEVICES)
    def test_mul_f32(self, device):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([2.0, 3.0, 0.5, -1.0], dtype=np.float32)
        ta, tb = make_vnn(a, F32, device), make_vnn(b, F32, device)
        result = (ta * tb).to_numpy().flatten()
        expected = a * b
        assert np.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_sub_f32(self, device):
        a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = (make_vnn(a, F32, device) - make_vnn(b, F32, device)).to_numpy().flatten()
        assert np.allclose(result, a - b, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_div_f32(self, device):
        a = np.array([6.0, 4.0, 9.0], dtype=np.float32)
        b = np.array([2.0, 2.0, 3.0], dtype=np.float32)
        result = (make_vnn(a, F32, device) / make_vnn(b, F32, device)).to_numpy().flatten()
        assert np.allclose(result, a / b, atol=1e-5)

    def test_mul_scalar(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = make_vnn(data, F32, "cpu").mul_scalar(3.0).to_numpy().flatten()
        assert np.allclose(result, data * 3.0)


# ===========================================================================
# GROUP 2: Shape ops
# ===========================================================================

class TestShapeOps:
    def test_reshape_roundtrip(self):
        data = np.arange(12, dtype=np.float32)
        t = make_vnn(data, F32, "cpu")
        t3x4 = t.reshape([3, 4])
        assert t3x4.shape == [3, 4]
        tback = t3x4.reshape([12])
        assert np.allclose(tback.to_numpy().flatten(), data)

    def test_squeeze_unsqueeze(self):
        data = np.ones((1, 4, 1), dtype=np.float32)
        t = make_vnn(data.reshape(4), F32, "cpu").unsqueeze(0).unsqueeze(2)
        assert t.shape == [1, 4, 1]
        squeezed = t.squeeze()
        assert squeezed.shape == [4]

    def test_flatten(self):
        data = np.ones((2, 3, 4), dtype=np.float32)
        t = make_vnn(data.flatten(), F32, "cpu").reshape([2, 3, 4])
        flat = t.flatten(0, 2)
        assert flat.shape == [24]


# ===========================================================================
# GROUP 3: Activations (GELU / LeakyReLU / ELU / Tanh / Clamp)
# ===========================================================================

class TestActivations:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gelu_parity(self, device):
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = F.gelu(torch.tensor(data)).numpy()
        result = make_vnn(data, F32, device).gelu().to_numpy().flatten()
        assert np.allclose(result, expected, atol=2e-3)

    def test_leaky_relu_parity(self):
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = F.leaky_relu(torch.tensor(data), negative_slope=0.01).numpy()
        result = make_vnn(data, F32, "cpu").leaky_relu(0.01).to_numpy().flatten()
        assert np.allclose(result, expected, atol=1e-5)

    def test_elu_parity(self):
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = F.elu(torch.tensor(data), alpha=1.0).numpy()
        result = make_vnn(data, F32, "cpu").elu(1.0).to_numpy().flatten()
        assert np.allclose(result, expected, atol=1e-5)

    def test_tanh_parity(self):
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = np.tanh(data)
        result = make_vnn(data, F32, "cpu").tanh().to_numpy().flatten()
        assert np.allclose(result, expected, atol=1e-5)

    def test_clamp_parity(self):
        data = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
        expected = np.clip(data, -2.0, 2.0)
        result = make_vnn(data, F32, "cpu").clamp(-2.0, 2.0).to_numpy().flatten()
        assert np.allclose(result, expected, atol=1e-5)


# ===========================================================================
# GROUP 4: Reductions (sum / mean / max / min)
# ===========================================================================

class TestReductions:
    @pytest.mark.parametrize("device", DEVICES)
    def test_sum_full(self, device):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = make_vnn(data, F32, device).sum().to_numpy().flatten()[0]
        assert abs(result - 10.0) < 1e-4

    @pytest.mark.parametrize("device", DEVICES)
    def test_sum_dim0(self, device):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = data.sum(axis=0)
        result = make_vnn(data, F32, device).sum(dim=0).to_numpy()
        assert np.allclose(result, expected, atol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_sum_dim1(self, device):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = data.sum(axis=1)
        result = make_vnn(data, F32, device).sum(dim=1).to_numpy()
        assert np.allclose(result, expected, atol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_mean_full(self, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = make_vnn(data, F32, device).mean().to_numpy().flatten()[0]
        assert abs(result - 2.5) < 1e-4

    @pytest.mark.parametrize("device", DEVICES)
    def test_max_val(self, device):
        data = np.array([-1.0, 5.0, 3.0, 2.0], dtype=np.float32)
        result = make_vnn(data, F32, device).max_val().to_numpy().flatten()[0]
        assert abs(result - 5.0) < 1e-4

    @pytest.mark.parametrize("device", DEVICES)
    def test_min_val(self, device):
        data = np.array([-1.0, 5.0, 3.0, 2.0], dtype=np.float32)
        result = make_vnn(data, F32, device).min_val().to_numpy().flatten()[0]
        assert abs(result - (-1.0)) < 1e-4


# ===========================================================================
# GROUP 5: Softmax / Log-Softmax
# ===========================================================================

class TestSoftmax:
    @pytest.mark.xfail(reason="softmax not yet implemented", strict=False)
    @pytest.mark.parametrize("device", DEVICES)
    def test_softmax_sums_to_one(self, device):
        data = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]], dtype=np.float32)
        result = make_vnn(data, F32, device).softmax(dim=1).to_numpy()
        row_sums = result.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    @pytest.mark.xfail(reason="softmax not yet implemented", strict=False)
    def test_softmax_parity_torch(self):
        data = np.random.randn(4, 8).astype(np.float32)
        expected = torch.softmax(torch.tensor(data), dim=1).numpy()
        result = make_vnn(data, F32, "cpu").softmax(dim=1).to_numpy()
        assert np.allclose(result, expected, atol=1e-5)

    @pytest.mark.xfail(reason="log_softmax not yet implemented", strict=False)
    def test_log_softmax_non_positive(self):
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = make_vnn(data, F32, "cpu").log_softmax(dim=1).to_numpy()
        assert np.all(result <= 0.0)


# ===========================================================================
# GROUP 6: Tensor Creators
# ===========================================================================

class TestCreators:
    @pytest.mark.xfail(reason="Tensor.zeros not yet implemented", strict=False)
    @pytest.mark.parametrize("device", ["cpu", "vulkan"])
    def test_zeros(self, device):
        t = Tensor.zeros([3, 4], F32, device)
        assert t.shape == [3, 4]
        assert np.allclose(t.to_numpy(), 0.0)

    @pytest.mark.xfail(reason="Tensor.ones not yet implemented", strict=False)
    def test_ones(self):
        t = Tensor.ones([5], F32, "cpu")
        assert np.allclose(t.to_numpy(), 1.0)

    @pytest.mark.xfail(reason="Tensor.full not yet implemented", strict=False)
    def test_full(self):
        t = Tensor.full([2, 3], 7.0, F32, "cpu")
        assert np.allclose(t.to_numpy(), 7.0)

    @pytest.mark.xfail(reason="Tensor.rand not yet implemented", strict=False)
    def test_rand_range(self):
        t = Tensor.rand([1000], F32, "cpu")
        a = t.to_numpy().flatten()
        assert np.all(a >= 0.0) and np.all(a < 1.0)

    @pytest.mark.xfail(reason="Tensor.randn not yet implemented", strict=False)
    def test_randn_statistics(self):
        t = Tensor.randn([100_000], F32, "cpu")
        a = t.to_numpy().flatten()
        assert abs(a.mean()) < 0.02
        assert abs(a.std() - 1.0) < 0.02


if __name__ == "__main__":
    # Quick smoke test without pytest
    print("Running ReLU smoke test...")
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1_000_000).astype(np.float32)
    for dev in DEVICES:
        t = make_vnn(data, F32, dev)
        r = t.relu().to_numpy().flatten()
        exp = np.maximum(data, 0.0)
        ok = np.allclose(r, exp, atol=1e-5)
        print(f"  ReLU({dev}): {'✅ OK' if ok else '❌ FAIL'}")

# --- Softmax & Log-Softmax ---
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype_setup", DTYPES)
@pytest.mark.parametrize("shape, dim", [
    ((10,), 0),
    ((10, 20), 1),
    ((10, 20), -1),
    ((10, 20), 0),
    ((2, 3, 4), 2),
    ((2, 3, 4), 1),
    ((2, 3, 4), -1),
])
def test_softmax(dev, dtype_setup, shape, dim):
    vnn_dt, np_dt, tc_dt, tol = dtype_setup
    rng = np.random.default_rng(42)
    # Scaled down to prevent fp16 overflow during exp if values get too large
    # but actual softmax impl subtracts max first so overflow is avoided anyway.
    data = (rng.standard_normal(shape) * 5.0).astype(np_dt)
    
    t_vnn = make_vnn(data, vnn_dt, dev)
    r_vnn = t_vnn.softmax(dim).to_numpy()
    
    t_tc = torch.from_numpy(data)
    r_tc = torch.nn.functional.softmax(t_tc, dim=dim).numpy()
    
    # Softmax output validation: sums to 1.0 along dim
    sums_vnn = np.sum(r_vnn, axis=dim)
    np.testing.assert_allclose(sums_vnn, np.ones_like(sums_vnn), rtol=tol, atol=tol)
    
    np.testing.assert_allclose(r_vnn, r_tc, rtol=tol, atol=tol)

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype_setup", DTYPES)
@pytest.mark.parametrize("shape, dim", [
    ((10, 20), -1),
    ((10, 20), 0),
    ((2, 3, 4), 1),
])
def test_log_softmax(dev, dtype_setup, shape, dim):
    vnn_dt, np_dt, tc_dt, tol = dtype_setup
    rng = np.random.default_rng(43)
    data = (rng.standard_normal(shape) * 10.0).astype(np_dt)
    
    t_vnn = make_vnn(data, vnn_dt, dev)
    r_vnn = t_vnn.log_softmax(dim).to_numpy()
    
    t_tc = torch.from_numpy(data)
    r_tc = torch.nn.functional.log_softmax(t_tc, dim=dim).numpy()
    
    np.testing.assert_allclose(r_vnn, r_tc, rtol=tol, atol=tol)


# ===========================================================================
# GROUP 6: Creators (zeros, ones, full, rand, randn)
# ===========================================================================

class TestCreators:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_setup", DTYPES)
    def test_zeros(self, device, dtype_setup):
        vnn_dt, np_dt, tc_dt, tol = dtype_setup
        shape = (10, 20)
        t = Tensor.zeros(shape, dtype=vnn_dt, device=device)
        assert t.shape == list(shape)
        assert t.device == device
        assert np.all(t.to_numpy() == 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_setup", DTYPES)
    def test_ones(self, device, dtype_setup):
        vnn_dt, np_dt, tc_dt, tol = dtype_setup
        shape = (4, 4, 4)
        t = Tensor.ones(shape, dtype=vnn_dt, device=device)
        assert t.shape == list(shape)
        assert t.device == device
        assert np.all(t.to_numpy() == 1.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_setup", DTYPES)
    def test_full(self, device, dtype_setup):
        vnn_dt, np_dt, tc_dt, tol = dtype_setup
        shape = (5, 5)
        # using integer or explicitly checking bounds
        val = 3.0
        t = Tensor.full(shape, val, dtype=vnn_dt, device=device)
        assert t.shape == list(shape)
        assert t.device == device
        assert np.allclose(t.to_numpy(), val, atol=tol)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_setup", DTYPES)
    def test_rand(self, device, dtype_setup):
        vnn_dt, np_dt, tc_dt, tol = dtype_setup
        shape = (1000,)
        t = Tensor.rand(shape, dtype=vnn_dt, device=device)
        arr = t.to_numpy()
        assert t.shape == list(shape)
        assert t.device == device
        assert np.all((arr >= 0.0) & (arr <= 1.0))
        assert np.mean(arr) > 0.4 and np.mean(arr) < 0.6

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_setup", DTYPES)
    def test_randn(self, device, dtype_setup):
        vnn_dt, np_dt, tc_dt, tol = dtype_setup
        shape = (10000,)
        t = Tensor.randn(shape, dtype=vnn_dt, device=device)
        arr = t.to_numpy()
        assert t.shape == list(shape)
        assert t.device == device
        # Standard normal distribution checks
        assert abs(np.mean(arr)) < 0.1
        assert 0.8 < np.std(arr) < 1.2
