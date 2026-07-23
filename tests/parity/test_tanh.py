"""Parity: oxtorch tanh (f32) vs f64 oracle, np.tanh, torch.tanh (Wave 2).

Cephes tanhf two-branch (small-x poly + large-x via exp). Binding gate is
parity-vs-oracle <=2 ULP (Rust oracle_test proves 1-2 ULP for scalar/sse2/avx1).
np.tanh is the special-function cross-check (tanh IS elementary but np.tanh in
f64 is the accurate reference).
"""

import numpy as np
import torch
import vulkannn_rusted as vnn


def _ox(x):
    return np.asarray(vnn.cpu_unary_f32("tanh", x.astype(np.float32).tolist()), dtype=np.float32)


def _bit_ulp(a, b):
    fin = np.isfinite(a) & np.isfinite(b)
    return int(np.max(np.abs(a[fin].view(np.int32) - b[fin].view(np.int32)))) if fin.any() else 0


def test_tanh_vs_f64_oracle_2ulp():
    x = np.linspace(-20, 20, 60_000, dtype=np.float32)
    oracle = np.tanh(x.astype(np.float64)).astype(np.float32)
    assert _bit_ulp(_ox(x), oracle) <= 2


def test_tanh_vs_torch():
    x = np.linspace(-20, 20, 30_000, dtype=np.float32)
    want = torch.tanh(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(_ox(x), want, rtol=1e-6, atol=1e-7)


def test_tanh_edge_cases():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 50.0, -50.0, 0.625, -0.625], np.float32)
    got = _ox(x)
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] == 1.0 and got[3] == -1.0
    assert np.isnan(got[4])
    assert got[5] == 1.0 and got[6] == -1.0
