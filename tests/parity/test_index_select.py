"""Parity: oxtorch index_select vs torch.index_select (Wave 4).

Pure gather (per-row memcpy) -> BITWISE equality criterion, no tolerance.
Covers: bit-exact gather, DUPLICATE indices (valid, common in embeddings), and
OUT-OF-RANGE (torch errors; oxtorch validates up front and raises, NOT UB —
legacy did an unchecked pointer add).
"""

import numpy as np
import pytest
import torch
import vulkannn_rusted as vnn

from conftest import to_oxtorch


def _isel(w, dtype, dim, idx):
    wt = to_oxtorch(w, dtype)
    it = to_oxtorch(idx.astype(np.float32), "f32")
    return np.asarray(wt.index_select(dim, it).to_numpy(), dtype=np.float32)


def test_index_select_f32_vs_torch():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((10, 4)).astype(np.float32)
    idx = np.array([3, 0, 9, 1, 5], np.int64)
    got = _isel(w, "f32", 0, idx)
    want = torch.index_select(torch.from_numpy(w), 0, torch.from_numpy(idx)).numpy()
    np.testing.assert_array_equal(got.ravel(), want.ravel())


def test_index_select_duplicates():
    w = np.arange(8, dtype=np.float32).reshape(4, 2)
    idx = np.array([2, 0, 2, 2, 3], np.int64)
    got = _isel(w, "f32", 0, idx)
    want = torch.index_select(torch.from_numpy(w), 0, torch.from_numpy(idx)).numpy()
    np.testing.assert_array_equal(got.ravel(), want.ravel())


def test_index_select_dtypes():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((6, 5)).astype(np.float32)
    idx = np.array([0, 5, 2, 2], np.int64)
    for dtype in ("f16", "bf16"):
        got = _isel(w, dtype, 0, idx)
        wr = to_oxtorch(w, dtype).to_numpy().astype(np.float32)
        want = wr[idx]
        np.testing.assert_array_equal(got.ravel(), want.ravel())


def test_index_select_out_of_range_raises():
    # torch raises; oxtorch validates and raises (a Rust panic surfaced by pyo3),
    # rather than the legacy unchecked pointer add (UB).
    w = np.arange(8, dtype=np.float32).reshape(4, 2)
    with pytest.raises(BaseException):
        _isel(w, "f32", 0, np.array([7], np.int64))  # row 7 in a 4-row table
