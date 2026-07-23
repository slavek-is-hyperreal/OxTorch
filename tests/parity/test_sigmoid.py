"""Parity: oxtorch sigmoid (f32) vs f64 oracle, scipy.special.expit, torch.

sigmoid IS a scipy special function (expit), so the addendum's scipy cross-check
applies here. Binding gate is parity-vs-oracle <=2 ULP (Rust oracle_test proves
2 ULP for scalar/sse2/avx1); this mirrors it in Python. Numerically-stable form
(exp on -|x|) so the large-negative denormal tail is preserved, not flushed.
"""

import numpy as np
import scipy.special as sp
import torch
import vulkannn_rusted as vnn


def _ox(x):
    return np.asarray(vnn.cpu_unary_f32("sigmoid", x.astype(np.float32).tolist()), dtype=np.float32)


def _bit_ulp(a, b):
    fin = np.isfinite(a) & np.isfinite(b)
    return int(np.max(np.abs(a[fin].view(np.int32) - b[fin].view(np.int32)))) if fin.any() else 0


def test_sigmoid_vs_f64_oracle_2ulp():
    x = np.linspace(-40, 40, 50_000, dtype=np.float32)
    oracle = (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    assert _bit_ulp(_ox(x), oracle) <= 2


def test_sigmoid_vs_scipy_expit():
    x = np.linspace(-40, 40, 20_000, dtype=np.float32)
    want = sp.expit(x.astype(np.float64)).astype(np.float32)
    assert _bit_ulp(_ox(x), want) <= 2


def test_sigmoid_vs_torch():
    x = np.linspace(-40, 40, 20_000, dtype=np.float32)
    want = torch.sigmoid(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(_ox(x), want, rtol=1e-6, atol=1e-7)


def test_sigmoid_edge_cases():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 100.0, -100.0], np.float32)
    got = _ox(x)
    assert got[0] == 0.5 and got[1] == 0.5
    assert got[2] == 1.0
    assert got[3] == 0.0
    assert np.isnan(got[4])
    assert got[5] == 1.0
    # sigmoid(-100) ~ exp(-100) ~ 3.8e-44 (denormal), NOT flushed to 0
    assert 0.0 < got[6] < 1e-42
