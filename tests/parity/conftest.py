"""Parity fixtures: oxtorch CPU kernels vs. PyTorch.

This is the Python half of the Wave-0 verification contract. The Rust half
(`vulkannn_rusted/src/cpu/parity_harness.rs`) proves a migrated kernel matches
the *legacy* kernel it replaces; this file proves the whole stack matches
*PyTorch*, which is the behaviour users actually depend on.

Usage from a test module in this directory::

    def test_sub_f32(parity):
        parity("sub", "f32", shapes=[(1024,), (37,)], tol=0.0)

or, without the fixture::

    from conftest import assert_parity

Every call additionally sweeps :data:`MANDATORY_LENS` — 1-D lengths chosen so
that ``n % vector_width != 0`` for every SIMD width and unroll factor in the
repo — and seeds each input with :data:`SPECIALS` (±0, ±inf, NaN, denormals).
You therefore do not need to (and should not) re-specify those per test.

Extending: add an entry to :data:`BINARY_OPS` / :data:`UNARY_OPS` and, for a new
element type, to :data:`DTYPES`. Nothing else in this file should need editing.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pytest
import torch

import vulkannn_rusted as vnn

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DTypeSpec:
    """How one element type is built, compared and named."""

    vnn_dtype: object
    torch_dtype: torch.dtype
    #: Default absolute tolerance. Exact for f32 (identical IEEE ops); loose for
    #: the reduced-precision types, where oxtorch computes in f32 and rounds
    #: once, while torch may round differently.
    default_tol: float
    #: True when a value must survive a lossy round-trip through the dtype
    #: before it can be used as a reference input.
    lossy: bool


DTYPES: dict[str, DTypeSpec] = {
    "f32": DTypeSpec(vnn.DataType.F32, torch.float32, default_tol=0.0, lossy=False),
    "f16": DTypeSpec(vnn.DataType.F16, torch.float16, default_tol=1e-3, lossy=True),
    "bf16": DTypeSpec(vnn.DataType.BF16, torch.bfloat16, default_tol=1e-2, lossy=True),
}

# ---------------------------------------------------------------------------
# operation registry
# ---------------------------------------------------------------------------

#: name -> (oxtorch callable, torch callable)
BINARY_OPS: dict[str, tuple[Callable, Callable]] = {
    "add": (operator.add, torch.add),
    "sub": (operator.sub, torch.sub),
    "mul": (operator.mul, torch.mul),
    "div": (operator.truediv, torch.div),
    "atan2": (lambda a, b: a.atan2(b), torch.atan2),
}

UNARY_OPS: dict[str, tuple[Callable, Callable]] = {
    "neg": (lambda a: a.neg(), torch.neg),
    "relu": (lambda a: a.relu(), torch.relu),
    "sigmoid": (lambda a: a.sigmoid(), torch.sigmoid),
    "silu": (lambda a: a.silu(), torch.nn.functional.silu),
    "tanh": (lambda a: a.tanh(), torch.tanh),
    "gelu": (lambda a: a.gelu(), torch.nn.functional.gelu),
}

# ---------------------------------------------------------------------------
# data generation — mirrors vulkannn_rusted/src/cpu/parity_harness.rs
# ---------------------------------------------------------------------------

#: Values every input vector starts with, when it is long enough to hold them.
SPECIALS: tuple[float, ...] = (
    0.0,
    -0.0,
    np.inf,
    -np.inf,
    np.nan,
    -np.nan,
    np.finfo(np.float32).tiny,  # smallest normal
    -np.finfo(np.float32).tiny,
    1e-40,  # denormal
    -1e-40,
    1.4012985e-45,  # f32 bit pattern 0x00000001, smallest denormal
    -1.4012985e-45,
    np.finfo(np.float32).max,
    np.finfo(np.float32).min,
    1.0,
    -1.0,
)

#: Lengths every parity call covers on top of the caller's shapes.
MANDATORY_LENS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025,
)


def make_data(shape: Sequence[int], seed: int, rotate: int = 0) -> np.ndarray:
    """Deterministic f32 input: specials first, seeded noise after.

    Deterministic on purpose — a flaky parity failure that cannot be reproduced
    is worse than no test at all.
    """
    n = int(np.prod(shape)) if len(shape) else 1
    flat = np.empty(n, dtype=np.float32)
    head = min(n, len(SPECIALS))
    for i in range(head):
        flat[i] = SPECIALS[(i + rotate) % len(SPECIALS)]
    if n > head:
        rng = np.random.default_rng(seed)
        flat[head:] = rng.uniform(-8.0, 8.0, size=n - head).astype(np.float32)
    return flat.reshape(shape)


# ---------------------------------------------------------------------------
# tensor bridging
# ---------------------------------------------------------------------------


def to_oxtorch(data: np.ndarray, dtype: str, device: str = "cpu"):
    """Build an oxtorch Tensor from a float32 numpy array.

    Note the constructor quirk: passing both ``shape`` and ``data`` makes
    oxtorch reinterpret the buffer bytewise. Always pass ``data`` alone and let
    the shape be inferred.
    """
    spec = DTYPES[dtype]
    arr = np.ascontiguousarray(data, dtype=np.float32)
    return vnn.Tensor(data=arr, dtype=spec.vnn_dtype, device=device)


def to_torch(data: np.ndarray, dtype: str) -> torch.Tensor:
    spec = DTYPES[dtype]
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(spec.torch_dtype)


def _as_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _compare(got: np.ndarray, want: np.ndarray, *, tol: float, ctx: str) -> None:
    got = np.ascontiguousarray(got, dtype=np.float32)
    want = np.ascontiguousarray(want, dtype=np.float32)

    if got.shape != want.shape:
        raise AssertionError(f"{ctx}: shape {got.shape} != torch {want.shape}")

    g = got.ravel()
    w = want.ravel()

    # NaN-ness must agree exactly; NaN payload/sign may differ (SIMD vs scalar).
    g_nan, w_nan = np.isnan(g), np.isnan(w)
    if not np.array_equal(g_nan, w_nan):
        bad = int(np.argmax(g_nan != w_nan))
        raise AssertionError(
            f"{ctx}: NaN-ness differs at index {bad}: oxtorch={g[bad]!r} torch={w[bad]!r}"
        )

    finite_mask = ~g_nan
    gf, wf = g[finite_mask], w[finite_mask]
    if gf.size == 0:
        return

    if tol == 0.0:
        # Bit-exact, so +0.0 vs -0.0 and infinity signs are both caught.
        g_bits = np.ascontiguousarray(gf).view(np.uint32)
        w_bits = np.ascontiguousarray(wf).view(np.uint32)
        if not np.array_equal(g_bits, w_bits):
            bad = int(np.argmax(g_bits != w_bits))
            raise AssertionError(
                f"{ctx}: not bit-identical at finite index {bad}: "
                f"oxtorch={gf[bad]!r} torch={wf[bad]!r}"
            )
        return

    np.testing.assert_allclose(gf, wf, atol=tol, rtol=tol, equal_nan=True, err_msg=ctx)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _normalise_shapes(shapes: Iterable | None) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [(n,) for n in MANDATORY_LENS]
    for s in shapes or ():
        out.append((s,) if isinstance(s, int) else tuple(s))
    # dedupe, preserving order
    seen: set[tuple[int, ...]] = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def assert_parity(op_name: str, dtype: str, shapes=None, tol: float | None = None) -> None:
    """Assert an oxtorch op matches PyTorch for `dtype` across `shapes`.

    Args:
        op_name: key in :data:`BINARY_OPS` or :data:`UNARY_OPS` (e.g. ``"sub"``).
        dtype: key in :data:`DTYPES` (``"f32"``, ``"f16"``, ``"bf16"``).
        shapes: extra shapes to cover, as ints or tuples. :data:`MANDATORY_LENS`
            is always included on top.
        tol: absolute+relative tolerance. ``0.0`` demands bit-exact equality.
            Defaults to the dtype's :attr:`DTypeSpec.default_tol`.

    Raises:
        AssertionError: on any mismatch, naming the op, dtype, shape and index.
    """
    if dtype not in DTYPES:
        raise KeyError(f"unknown dtype {dtype!r}; known: {sorted(DTYPES)}")
    spec = DTYPES[dtype]
    tol = spec.default_tol if tol is None else tol

    if op_name in BINARY_OPS:
        arity, (ox_fn, th_fn) = 2, BINARY_OPS[op_name]
    elif op_name in UNARY_OPS:
        arity, (ox_fn, th_fn) = 1, UNARY_OPS[op_name]
    else:
        raise KeyError(
            f"unknown op {op_name!r}; known: {sorted(BINARY_OPS)} / {sorted(UNARY_OPS)}"
        )

    for shape in _normalise_shapes(shapes):
        ctx = f"{op_name}/{dtype}/shape={shape}"

        a_np = make_data(shape, seed=0x9E3779B9, rotate=0)
        ox_args = [to_oxtorch(a_np, dtype)]
        th_args = [to_torch(a_np, dtype)]
        if arity == 2:
            b_np = make_data(shape, seed=0x85EBCA6B, rotate=3)
            ox_args.append(to_oxtorch(b_np, dtype))
            th_args.append(to_torch(b_np, dtype))

        got = ox_fn(*ox_args).to_numpy()
        want = _as_f32(th_fn(*th_args))
        _compare(np.asarray(got, dtype=np.float32), want, tol=tol, ctx=ctx)


@pytest.fixture
def parity():
    """Fixture wrapper around :func:`assert_parity`."""
    return assert_parity


@pytest.fixture
def make_parity_data():
    """Fixture wrapper around :func:`make_data`, for hand-rolled comparisons."""
    return make_data
