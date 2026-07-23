"""Parity: oxtorch matmul kernels vs PyTorch (Wave 5).

matmul is a MOVE-NOT-REWRITE migration: cpu/ops/matmul/ is a byte-for-byte
relocation of cpu_old/ops/matmul/, backed by the `matrixmultiply` crate. There
is therefore no Rust golden-vs-legacy test (the two are the same code); this
file proves the whole stack matches torch, which is what users depend on.

Python reaches the kernel via `Tensor.__matmul__` (the `@` operator), which
dispatches to matmul_f32 / matmul_f16 (and bf16 through the f32 fallback path).
Shapes deliberately cross the TILE_SIZE=256 boundary and use non-tile-aligned,
non-square dims so the f16/bf16 tiling + remainder handling is exercised.

f32 is compared against a matrixmultiply-vs-torch tolerance rather than exact:
both accumulate in f32 but block/tile in different orders, so the low bits of a
long-K reduction legitimately differ. f16/bf16 additionally round the output.
"""

import numpy as np
import pytest
import torch

import vulkannn_rusted as vnn

from conftest import to_oxtorch, to_torch, _as_f32, make_data


# (M, K, N): small; K spanning one and several 256-tiles; non-aligned; skinny.
SHAPES = [
    (2, 3, 4),
    (33, 31, 17),
    (64, 64, 64),
    (128, 300, 96),      # K > TILE_SIZE (one full tile + remainder)
    (300, 512, 257),     # M, K, N all cross tile boundaries
    (1, 1024, 1),        # long-K vector outer product edge
]


def _mm_data(m, k, n, seed):
    a = make_data((m, k), seed).astype(np.float32)
    b = make_data((k, n), seed + 1).astype(np.float32)
    # matmul over ±inf/NaN specials makes the reference ill-defined; keep it
    # finite so the comparison tests the arithmetic, not IEEE propagation (that
    # is covered by the elementwise parity suites).
    a = np.nan_to_num(a, nan=0.3, posinf=2.0, neginf=-2.0)
    b = np.nan_to_num(b, nan=-0.4, posinf=1.5, neginf=-1.5)
    # Clip to a modest range: make_data seeds finfo.max/min at the head, which
    # overflows an f32 accumulation of length K to inf and makes the reference
    # ill-defined. Overflow/IEEE propagation is the elementwise suites' job; here
    # we test the GEMM arithmetic on well-scaled inputs.
    return np.clip(a, -4.0, 4.0), np.clip(b, -4.0, 4.0)


def _ox_to_np(t):
    return np.asarray(t.to_numpy(), dtype=np.float32)


@pytest.mark.parametrize("shape", SHAPES)
def test_matmul_f32(shape):
    m, k, n = shape
    a, b = _mm_data(m, k, n, seed=7 + m + k + n)
    oa, ob = to_oxtorch(a, "f32"), to_oxtorch(b, "f32")
    got = _ox_to_np(oa @ ob)
    want = _as_f32(to_torch(a, "f32") @ to_torch(b, "f32"))
    # matrixmultiply vs torch: both f32-accumulate, different blocking order.
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", SHAPES)
def test_matmul_f16(shape):
    m, k, n = shape
    a, b = _mm_data(m, k, n, seed=9 + m + k + n)
    oa, ob = to_oxtorch(a, "f16"), to_oxtorch(b, "f16")
    got = _ox_to_np(oa @ ob)
    # oxtorch upcasts f16->f32, matmuls in f32, rounds once. Reference mirrors
    # that (compute in f32) so we test the kernel, not torch's f16 accumulator.
    a16 = _as_f32(to_torch(a, "f16"))
    b16 = _as_f32(to_torch(b, "f16"))
    want = a16 @ b16
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-2 * max(1, k))


@pytest.mark.parametrize("shape", SHAPES)
def test_matmul_bf16(shape):
    m, k, n = shape
    a, b = _mm_data(m, k, n, seed=11 + m + k + n)
    oa, ob = to_oxtorch(a, "bf16"), to_oxtorch(b, "bf16")
    got = _ox_to_np(oa @ ob)
    a16 = _as_f32(to_torch(a, "bf16"))
    b16 = _as_f32(to_torch(b, "bf16"))
    want = a16 @ b16
    np.testing.assert_allclose(got, want, rtol=5e-2, atol=5e-2 * max(1, k))
