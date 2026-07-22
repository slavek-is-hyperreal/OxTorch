"""Parity: oxtorch div kernels vs PyTorch (Wave 1).

div can't use the generic `parity` fixture: legacy div guards `b == 0` and
returns 0.0 (scalar) while its SIMD body divides raw (±inf), a documented
oxtorch-vs-torch divergence preserved verbatim under Rule 6. So we test the two
regimes separately:

  * well-defined case (nonzero denominator) — must match torch bit-for-bit;
  * the /0 legacy quirk — pinned to the scalar path via force_arch, asserting the
    guarded 0.0 result (NOT torch's inf). This documents the semantic; it is not
    a claim that oxtorch matches torch on /0.

The Rust golden test (cpu/ops/binary/div/mod.rs::parity) already proves the
migrated kernels match legacy bit-for-bit including the quirk.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import DTYPES, make_data, to_oxtorch, to_torch, _as_f32, _compare


def _finite(shape, seed, *, nonzero=False):
    """Plain finite values, no specials. Div's special-input coverage lives in
    the Rust golden test (new vs legacy); here we only need well-defined inputs
    so oxtorch-vs-torch is unambiguous."""
    n = int(np.prod(shape)) if len(shape) else 1
    v = np.random.default_rng(seed).uniform(-8.0, 8.0, size=n).astype(np.float32)
    if nonzero:
        v = np.where(np.abs(v) < 0.5, np.copysign(0.5, v) + (v == 0), v)
    return v.reshape(shape)


def _check(dtype, shape, tol):
    ctx = f"div/{dtype}/shape={shape}"
    a_np = _finite(shape, seed=0x9E3779B9)
    b_np = _finite(shape, seed=0x85EBCA6B, nonzero=True)
    ox = to_oxtorch(a_np, dtype) / to_oxtorch(b_np, dtype)
    want = _as_f32(torch.div(to_torch(a_np, dtype), to_torch(b_np, dtype)))
    _compare(np.asarray(ox.to_numpy(), dtype=np.float32), want, tol=tol, ctx=ctx)


def test_div_f32_nonzero():
    for shape in [(1,), (7,), (8,), (17,), (1024,), (33, 31)]:
        _check("f32", shape, tol=0.0)  # IEEE divide both sides -> bit-exact


def test_div_f16_nonzero():
    for shape in [(1024,), (33, 31)]:
        _check("f16", shape, tol=DTYPES["f16"].default_tol)


def test_div_bf16_nonzero():
    for shape in [(1024,), (33, 31)]:
        _check("bf16", shape, tol=DTYPES["bf16"].default_tol)


def test_div_by_zero_is_legacy_quirk_zero():
    """Legacy semantic: scalar path returns 0.0 for x/0 (not torch's inf)."""
    vnn.force_arch("scalar")
    try:
        a = to_oxtorch(np.array([1.0, -3.0, 0.0, 5.0], np.float32), "f32")
        b = to_oxtorch(np.array([0.0, 0.0, 0.0, 2.0], np.float32), "f32")
        got = np.asarray((a / b).to_numpy(), dtype=np.float32)
    finally:
        vnn.force_arch("auto")
    assert np.array_equal(got, np.array([0.0, 0.0, 0.0, 2.5], np.float32)), got
