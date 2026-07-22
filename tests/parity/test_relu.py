"""Parity: oxtorch relu kernels vs PyTorch (Wave 2). max(x, 0), memory-bound.

NaN handling diverges by design: oxtorch relu(NaN)=0.0 (Rust `max` and
`_mm_max_ps` both return the non-NaN operand), torch relu(NaN)=NaN. This is
legacy behaviour preserved under Rule 6, so the vs-torch test uses finite/±inf
inputs; the NaN path and full special-input coverage live in the Rust golden
test (unary/relu/mod.rs::parity vs legacy). i8 is Rust-only too.
"""

import numpy as np
import torch
import vulkannn_rusted as vnn

from conftest import DTYPES, _as_f32


def _check(dtype, shape, tol):
    spec = DTYPES[dtype]
    n = int(np.prod(shape))
    # finite + ±inf, no NaN
    rng = np.random.default_rng(0x2468)
    x = rng.uniform(-8.0, 8.0, size=n).astype(np.float32)
    x[: min(2, n)] = [np.inf, -np.inf][: min(2, n)]
    t = vnn.Tensor(data=x.reshape(shape), dtype=spec.vnn_dtype, device="cpu")
    got = np.asarray(t.relu().to_numpy(), dtype=np.float32)
    want = _as_f32(torch.relu(torch.from_numpy(x.reshape(shape)).to(spec.torch_dtype)))
    np.testing.assert_allclose(got.ravel(), want.ravel(), atol=tol, rtol=tol,
                               err_msg=f"relu/{dtype}/{shape}")


def test_relu_f32():
    for shape in [(1,), (7,), (8,), (17,), (1024,), (33, 31)]:
        _check("f32", shape, tol=0.0)


def test_relu_f16():
    for shape in [(1024,), (33, 31)]:
        _check("f16", shape, tol=DTYPES["f16"].default_tol)


def test_relu_bf16():
    for shape in [(1024,), (33, 31)]:
        _check("bf16", shape, tol=DTYPES["bf16"].default_tol)
