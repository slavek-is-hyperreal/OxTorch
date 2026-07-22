"""Parity: oxtorch tensor⊕scalar broadcast vs PyTorch (Wave 1, W1d).

Exposed via Python operators: `t + s`, `t - s`, `t * s`, `t / s` (scalar float).
f32/f16/bf16 compared vs torch. i8 is NOT tested vs torch (torch int8 wraps,
oxtorch saturates); i8 scalar coverage is the Rust test (scalar/mod.rs::parity).
div-by-zero returns 0.0 by legacy design (Rule 6) — tested with nonzero scalars.
"""

import operator

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import DTYPES, _as_f32

OPS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}


def _check(dtype, op_name, scalar):
    spec = DTYPES[dtype]
    x = np.linspace(-4.0, 4.0, 1000, dtype=np.float32)
    t = vnn.Tensor(data=x, dtype=spec.vnn_dtype, device="cpu")
    got = np.asarray(OPS[op_name](t, scalar).to_numpy(), dtype=np.float32)
    tx = torch.from_numpy(x).to(spec.torch_dtype)
    want = _as_f32(OPS[op_name](tx, scalar))
    np.testing.assert_allclose(got, want, atol=max(spec.default_tol, 1e-6),
                               rtol=max(spec.default_tol, 1e-6),
                               err_msg=f"scalar {op_name} {dtype} s={scalar}")


def test_scalar_f32():
    for op in OPS:
        _check("f32", op, 2.5)


def test_scalar_f16():
    for op in OPS:
        _check("f16", op, 2.5)


def test_scalar_bf16():
    for op in OPS:
        _check("bf16", op, 2.5)
