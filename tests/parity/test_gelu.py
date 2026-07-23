"""Parity: oxtorch gelu (f32, tanh-approx) vs torch (Wave 2).

MUST use torch approximate='tanh' — the default (erf) would fail on a definition
mismatch, not a bug (docs/kernel_specs/README.md §4). No scipy special fn for the
tanh-approx gelu. Rust oracle_test proves 4 ULP (well-conditioned) + tail flush.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn


def _ox(x):
    return np.asarray(vnn.cpu_unary_f32("gelu", x.astype(np.float32).tolist()), dtype=np.float32)


def test_gelu_vs_torch_tanh():
    x = np.linspace(-10, 10, 40_000, dtype=np.float32)
    want = torch.nn.functional.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    np.testing.assert_allclose(_ox(x), want, rtol=1e-5, atol=1e-6)


def test_gelu_edge_cases_match_torch():
    x = np.array([0.0, np.inf, -np.inf, np.nan, -100.0, -10.0], np.float32)
    got = _ox(x)
    want = torch.nn.functional.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    assert np.array_equal(np.isnan(got), np.isnan(want))
    fin = ~np.isnan(got)
    np.testing.assert_allclose(got[fin], want[fin], rtol=1e-5, atol=1e-6)
