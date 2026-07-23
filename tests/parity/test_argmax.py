"""Parity: oxtorch argmax vs torch (Wave 3).

Tie-breaking MATCHES torch (first max index). NaN DIVERGES: argmax ignores NaN
(picks the max of the non-NaN values) while torch returns the NaN index
(docs/known_divergences.md §6). So the vs-torch test uses non-NaN data; the NaN
divergence is asserted separately.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import to_oxtorch


def _argmax(x, dtype, dim):
    return np.asarray(to_oxtorch(x, dtype).argmax(dim).to_numpy())


def test_argmax_f32_vs_torch_1d():
    rng = np.random.default_rng(0)
    for n in (1, 8, 17, 1000):
        x = (rng.standard_normal(n) * 10).astype(np.float32)
        got = int(_argmax(x, "f32", 0).reshape(-1)[0])
        want = int(torch.argmax(torch.from_numpy(x)).item())
        assert got == want, (n, got, want)


def test_argmax_2d_axis():
    x = np.array([[1, 5, 2], [9, 0, 4], [3, 3, 7]], np.float32)
    got = _argmax(x, "f32", 1).reshape(-1).astype(int)
    want = torch.argmax(torch.from_numpy(x), dim=1).numpy().astype(int)
    np.testing.assert_array_equal(got, want)


def test_argmax_tie_first_index_matches_torch():
    x = np.array([3.0, 1.0, 3.0, 3.0], np.float32)
    assert int(_argmax(x, "f32", 0).reshape(-1)[0]) == 0
    assert int(torch.argmax(torch.from_numpy(x)).item()) == 0


def test_argmax_nan_divergence():
    # oxtorch ignores NaN -> index 2; torch returns NaN index -> 1.
    x = np.array([1.0, np.nan, 3.0], np.float32)
    assert int(_argmax(x, "f32", 0).reshape(-1)[0]) == 2
