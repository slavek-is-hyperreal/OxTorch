"""Parity: oxtorch softmax / log-softmax vs torch (Wave 3).

Reuses the validated max (stabilisation) + exp core (<=2 ULP) + f64 sum
accumulator (denominator). Tolerance MEASURED, not assumed: f32 softmax vs torch
is max abs err ~3e-8, max rel err ~2.3e-7 (our f64 denominator is at least as
accurate as torch), so rtol 1e-5 / atol 1e-6 is comfortable.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import DTYPES, to_oxtorch


def _softmax(x, dtype, dim, is_log):
    t = to_oxtorch(x, dtype)
    r = t.apply_softmax(dim, is_log) if is_log else t.softmax(dim)
    return np.asarray(r.to_numpy(), dtype=np.float32)


def test_softmax_f32_vs_torch():
    rng = np.random.default_rng(0)
    for shape, dim in [((100,), 0), ((8, 100), 1), ((32, 17), 1)]:
        x = (rng.standard_normal(shape) * 3).astype(np.float32)
        got = _softmax(x, "f32", dim, False)
        want = torch.softmax(torch.from_numpy(x), dim=dim).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_log_softmax_f32_vs_torch():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((8, 100)) * 3).astype(np.float32)
    got = _softmax(x, "f32", 1, True)
    want = torch.log_softmax(torch.from_numpy(x), dim=1).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_softmax_f16_bf16_vs_torch():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((8, 64)) * 2).astype(np.float32)
    for dtype, tol in (("f16", 2e-3), ("bf16", 1e-2)):
        spec = DTYPES[dtype]
        got = _softmax(x, dtype, 1, False)
        want = torch.softmax(torch.from_numpy(x).to(spec.torch_dtype), dim=1).to(torch.float32).numpy()
        np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((16, 50)) * 5).astype(np.float32)
    got = _softmax(x, "f32", 1, False)
    np.testing.assert_allclose(got.sum(axis=1), np.ones(16), atol=1e-5)
