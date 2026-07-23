"""Parity: oxtorch sum (f64 accumulator) vs numpy-f64 oracle AND torch (Wave 3).

Rule 6 / Wave-3 policy: the accumulator is ALWAYS f64 (f32/f16/bf16) or i64 (i8),
downcast to output only at the end. The Rust oracle_test proves the accumulator
is exact-f64 (scalar) / <=1e-12 rel (avx1 SIMD reassociation).

Two references, per the Wave-3 spec:
  * numpy.sum(float64) — the accurate oracle. oxtorch (f64-accumulate, f32 output)
    matches it within f32 downcast epsilon.
  * torch.sum — torch accumulates differently (roughly f32 / pairwise), so its
    result is LESS accurate than our f64 accumulator; the tolerance vs torch is
    therefore LOOSER, not tighter. We do not demand bit-equality with torch.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import to_oxtorch


def _sum(x, dtype):
    t = to_oxtorch(x, dtype)
    return np.asarray(t.sum(None).to_numpy(), dtype=np.float64).reshape(-1)[0]


def test_sum_f32_vs_numpy_f64():
    rng = np.random.default_rng(0)
    for n in (1, 8, 17, 1000, 200_000):
        x = (rng.standard_normal(n) * 10).astype(np.float32)
        got = _sum(x, "f32")
        oracle = np.sum(x.astype(np.float64))
        # oxtorch downcasts to f32 at the end; compare at f32 relative epsilon.
        assert abs(got - oracle) <= 1e-5 * max(abs(oracle), 1.0) + 1e-4, (n, got, oracle)


def test_sum_f32_vs_torch_loose():
    rng = np.random.default_rng(1)
    for n in (1000, 200_000):
        x = (rng.standard_normal(n) * 10).astype(np.float32)
        got = _sum(x, "f32")
        want = torch.sum(torch.from_numpy(x)).item()
        # torch accumulates less precisely than our f64 accumulator -> loose tol.
        assert abs(got - want) <= 1e-3 * max(abs(want), 1.0) + 1e-2, (n, got, want)


def test_sum_f16_bf16_vs_numpy_f64():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(5000) * 4).astype(np.float32)
    for dtype, tol in (("f16", 5e-2), ("bf16", 2e-1)):
        # oracle: round input through the dtype (lossy), then f64 sum.
        xr = to_oxtorch(x, dtype).to_numpy().astype(np.float64)
        oracle = np.sum(xr)
        got = _sum(x, dtype)
        assert abs(got - oracle) <= tol * max(abs(oracle), 1.0) + tol, (dtype, got, oracle)
