"""Parity: oxtorch neg kernels vs PyTorch (Wave 2). -x, memory-bound.

Known divergence (docs/known_divergences.md §4): the SIMD tiers compute `0 - x`,
so `neg(+0.0) = +0.0` instead of torch's `-0.0` (sign of zero lost). Legacy
behaviour, preserved under Rule 6. We normalise signed zeros before comparing
(`x + 0.0` maps -0.0 -> +0.0); NaN propagation and the sign-of-zero quirk itself
are covered by the Rust golden (unary/neg/mod.rs::parity vs legacy).
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import DTYPES, make_data, to_oxtorch, to_torch, _as_f32, _compare


def _check(dtype, shape, tol):
    spec = DTYPES[dtype]
    a = make_data(shape, seed=0x9E3779B9, rotate=0)
    got = np.asarray(to_oxtorch(a, dtype).neg().to_numpy(), dtype=np.float32) + 0.0
    want = _as_f32(torch.neg(to_torch(a, dtype))) + 0.0  # +0.0 normalises -0.0
    _compare(got, want, tol=tol, ctx=f"neg/{dtype}/{shape}")


def test_neg_f32():
    for shape in [(1024,), (33, 31)]:
        _check("f32", shape, tol=0.0)


def test_neg_f16():
    for shape in [(1024,), (33, 31)]:
        _check("f16", shape, tol=DTYPES["f16"].default_tol)


def test_neg_bf16():
    for shape in [(1024,), (33, 31)]:
        _check("bf16", shape, tol=DTYPES["bf16"].default_tol)
