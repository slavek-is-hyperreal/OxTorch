"""Parity: oxtorch layer_norm / rms_norm / sub_layer_norm vs torch (Wave 3).

f32 accumulation (Rule 6: norm stays f32, unlike the sum reduction) + std sqrt
(hardware, no rsqrt approximation). f16/bf16 convert-through-f32. Measured f32 vs
torch ~2.4e-7 max err -> rtol 1e-5.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import DTYPES, to_oxtorch


def _ln(x, dtype, dim, w, b, eps=1e-5):
    spec = DTYPES[dtype]
    t = to_oxtorch(x, dtype)
    wt = to_oxtorch(w, dtype)
    bt = to_oxtorch(b, dtype)
    return np.asarray(t.layer_norm([dim], wt, bt, eps).to_numpy(), dtype=np.float32)


def test_layer_norm_f32_vs_torch():
    rng = np.random.default_rng(0)
    for shape in [(4, 8), (16, 128), (3, 33)]:
        d = shape[-1]
        x = rng.standard_normal(shape).astype(np.float32)
        w = (rng.standard_normal(d) * 0.5 + 1).astype(np.float32)
        b = (rng.standard_normal(d) * 0.1).astype(np.float32)
        got = _ln(x, "f32", d, w, b)
        want = torch.nn.functional.layer_norm(
            torch.from_numpy(x), [d], torch.from_numpy(w), torch.from_numpy(b), 1e-5
        ).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_layer_norm_f16_bf16():
    rng = np.random.default_rng(1)
    d = 64
    x = rng.standard_normal((8, d)).astype(np.float32)
    w = np.ones(d, np.float32)
    b = np.zeros(d, np.float32)
    want = torch.nn.functional.layer_norm(torch.from_numpy(x), [d], torch.ones(d), torch.zeros(d), 1e-5).numpy()
    for dtype, tol in (("f16", 2e-3), ("bf16", 1e-2)):
        got = _ln(x, dtype, d, w, b)
        np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def test_rms_norm_f32_vs_torch():
    rng = np.random.default_rng(2)
    d = 128
    x = rng.standard_normal((8, d)).astype(np.float32)
    w = (rng.standard_normal(d) * 0.3 + 1).astype(np.float32)
    got = np.asarray(to_oxtorch(x, "f32").rms_norm([d], to_oxtorch(w, "f32"), 1e-6).to_numpy(), np.float32)
    xt = torch.from_numpy(x)
    rms = torch.sqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6)
    want = (xt / rms * torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_sub_layer_norm_zeroes_mean():
    # SubLN subtracts the mean then normalises (no bias). Output rows are
    # zero-mean-centred before scaling; with w=1 each row should have ~0 mean.
    rng = np.random.default_rng(3)
    d = 64
    x = rng.standard_normal((8, d)).astype(np.float32)
    w = np.ones(d, np.float32)
    got = np.asarray(to_oxtorch(x, "f32").subln([d], to_oxtorch(w, "f32"), 1e-6).to_numpy(), np.float32)
    # normalised (x-mean)/std has ~unit variance and ~zero mean per row
    assert np.all(np.abs(got.mean(axis=1)) < 1e-4)
    np.testing.assert_allclose(got.std(axis=1), np.ones(8), rtol=0.05)
