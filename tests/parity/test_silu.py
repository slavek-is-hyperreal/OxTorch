"""Parity: oxtorch silu (f32) vs torch.nn.functional.silu (Wave 2).

silu is x/(1+exp(-x)); torch is the reference (it defines the deep-tail flush to
∓0 we match — see docs/kernel_specs/silu_spec.md). No scipy special fn for silu.
Rust oracle_test proves 2 ULP over |x|<=40 vs the f64 naive form + torch-matching
edges; this mirrors it against torch.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn


def _ox(x):
    return np.asarray(vnn.cpu_unary_f32("silu", x.astype(np.float32).tolist()), dtype=np.float32)


def test_silu_vs_torch():
    x = np.linspace(-40, 40, 40_000, dtype=np.float32)
    want = torch.nn.functional.silu(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(_ox(x), want, rtol=1e-6, atol=1e-7)


def test_silu_edge_cases_match_torch():
    x = np.array([0.0, np.inf, -np.inf, np.nan, -100.0, 100.0], np.float32)
    got = _ox(x)
    want = torch.nn.functional.silu(torch.from_numpy(x)).numpy()
    # NaN-ness agrees
    assert np.array_equal(np.isnan(got), np.isnan(want))
    fin = ~np.isnan(got)
    np.testing.assert_array_equal(got[fin], want[fin])
    # explicit: torch flushes silu(-100) to -0.0
    assert got[4] == 0.0  # value equals (sign of zero not asserted by ==)
