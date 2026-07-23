"""Parity: oxtorch exp (f32) vs the f64 oracle AND scipy AND torch (Wave 2).

Per docs/kernel_specs/exp_spec.md the binding gate is parity-vs-oracle (f64 exp
rounded to f32) at <=2 ULP; the Rust test (unary/exp/fp32/mod.rs::oracle_test)
already proves 1 ULP for scalar/sse2/avx1 over a dense domain sweep + edge cases.
This mirrors that at the Python level and cross-checks scipy.special (same Cephes
lineage) and torch. exp reached via the test hook `cpu_unary_f32` (the high-level
unary path is not exposed to Python for in-memory tensors).
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

# NOTE: plain exp is not a scipy.special function (there is no scipy.special.exp);
# its authoritative reference is np.exp computed in f64. The scipy.special
# cross-check applies to the *special* members of this family — sigmoid
# (scipy.special.expit) and gelu (scipy.special.erf) — and is exercised in
# their own test files.


def _ox(x):
    return np.asarray(vnn.cpu_unary_f32("exp", x.astype(np.float32).tolist()), dtype=np.float32)


def _bit_ulp(a, b):
    fin = np.isfinite(a) & np.isfinite(b)
    return int(np.max(np.abs(a[fin].view(np.int32) - b[fin].view(np.int32)))) if fin.any() else 0


def test_exp_vs_f64_oracle_2ulp():
    x = np.linspace(-103.0, 88.7, 50_000, dtype=np.float32)
    got = _ox(x)
    oracle = np.exp(x.astype(np.float64)).astype(np.float32)
    assert _bit_ulp(got, oracle) <= 2


def test_exp_vs_torch():
    x = np.linspace(-40, 40, 20_000, dtype=np.float32)
    got = _ox(x)
    want = torch.exp(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


def test_exp_edge_cases():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 100.0, -200.0], np.float32)
    got = _ox(x)
    assert got[0] == 1.0 and got[1] == 1.0            # exp(±0) = 1
    assert got[2] == np.inf                            # exp(+inf)
    assert got[3] == 0.0                               # exp(-inf)
    assert np.isnan(got[4])                            # exp(NaN)
    assert got[5] == np.inf                            # overflow
    assert got[6] == 0.0                               # underflow
