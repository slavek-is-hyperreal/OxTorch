"""Parity: oxtorch max reduction vs torch (Wave 3).

max ignores NaN (Rust `f32::max` / `_mm*_max_ps` return the non-NaN operand) —
legacy behaviour, transcribed under Rule 6; torch propagates NaN
(docs/known_divergences.md §5). So the vs-torch test uses finite/±inf data; the
NaN divergence is documented and asserted separately (oxtorch drops it).
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import to_oxtorch


def _max(x, dtype):
    return np.asarray(to_oxtorch(x, dtype).reduce("max", None).to_numpy(), dtype=np.float64).reshape(-1)[0]


def test_max_f32_vs_torch():
    rng = np.random.default_rng(0)
    for n in (1, 8, 17, 1000, 200_000):
        x = (rng.standard_normal(n) * 10).astype(np.float32)
        x[min(1, n - 1)] = np.inf if n > 3 else x[0]
        got = _max(x, "f32")
        want = torch.max(torch.from_numpy(x)).item()
        assert got == want, (n, got, want)


def test_max_f16_bf16_vs_torch():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(5000) * 4).astype(np.float32)
    for dtype in ("f16", "bf16"):
        xr = to_oxtorch(x, dtype).to_numpy().astype(np.float32)
        got = _max(x, dtype)
        want = float(np.max(xr))
        assert abs(got - want) <= 1e-2, (dtype, got, want)


def test_max_ignores_nan_divergence():
    # Documented: oxtorch max drops NaN (=3), torch would propagate (=nan).
    x = np.array([1.0, np.nan, 3.0], np.float32)
    assert _max(x, "f32") == 3.0
