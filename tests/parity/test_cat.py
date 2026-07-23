"""Parity: oxtorch cat vs torch.cat (Wave 4).

Pure memory movement (no numeric accumulation) -> criterion is BITWISE equality,
not a tolerance. A non-bit difference would be a layout/stride bug, not something
to paper over with atol. Edge shapes: single tensor (no-op), N tensors, inner
axes, an empty (size-0-on-axis) tensor.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import to_oxtorch


def _cat(arrs, dtype, dim):
    # cat is a static-style method taking the FULL tensor list.
    ts = [to_oxtorch(a, dtype) for a in arrs]
    r = vnn.Tensor.cat(ts, dim)
    return np.asarray(r.to_numpy(), dtype=np.float32)


def _check(arrs, dtype, dim):
    got = _cat(arrs, dtype, dim)
    want = torch.cat([torch.from_numpy(a) for a in arrs], dim=dim).numpy()
    np.testing.assert_array_equal(got.ravel(), want.ravel())
    assert got.shape == want.shape, (got.shape, want.shape)


def test_cat_dim0():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    _check([a, b], "f32", 0)


def test_cat_dim1_inner():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 2)).astype(np.float32)
    b = rng.standard_normal((3, 5)).astype(np.float32)
    _check([a, b], "f32", 1)


def test_cat_n_tensors():
    rng = np.random.default_rng(2)
    arrs = [rng.standard_normal((2, 4)).astype(np.float32) for _ in range(3)]
    _check(arrs, "f32", 0)


def test_cat_single_tensor_noop():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = _cat([a], "f32", 0)
    np.testing.assert_array_equal(got, a)


def test_cat_with_empty_on_axis():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    e = np.zeros((0, 3), dtype=np.float32)
    _check([a, e], "f32", 0)


def test_cat_dtypes():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((2, 3)).astype(np.float32)
    for dtype, tol in (("f16", 0), ("bf16", 0)):
        # bit-exact after both inputs are rounded to the dtype (cat only copies).
        got = _cat([a, b], dtype, 0)
        ar = to_oxtorch(a, dtype).to_numpy().astype(np.float32)
        br = to_oxtorch(b, dtype).to_numpy().astype(np.float32)
        want = np.concatenate([ar, br], axis=0)
        np.testing.assert_array_equal(got.ravel(), want.ravel())
