"""Parity: oxtorch pow (tensor ^ scalar) vs PyTorch (Wave 1, W1d).

pow was relocated from unary to binary/pow. exponent==2.0 hits the SIMD square
fast path; other exponents use scalar powf. Negative base with fractional
exponent yields NaN on both sides (compared NaN-aware).
"""

import numpy as np
import torch
import vulkannn_rusted as vnn


def _pow(x, exp):
    t = vnn.Tensor(data=x, dtype=vnn.DataType.F32, device="cpu")
    return np.asarray(t.pow(exp).to_numpy(), dtype=np.float32)


def test_pow_f32():
    x = np.linspace(-6.0, 6.0, 1000, dtype=np.float32)
    for exp in (2.0, 3.0, 0.5, -1.0, 1.0):
        got = _pow(x, exp)
        want = torch.pow(torch.from_numpy(x), exp).numpy()
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5, equal_nan=True,
                                   err_msg=f"pow exp={exp}")


def test_pow2_fast_path_exact():
    # exponent 2.0 is x*x on both sides -> bit-exact.
    x = np.linspace(-6.0, 6.0, 1023, dtype=np.float32)  # prime-ish: exercises tail
    got = _pow(x, 2.0)
    want = (x * x).astype(np.float32)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
