"""Parity: oxtorch sub kernels vs PyTorch.

f32/bf16 migrated in Wave 0, f16 in Wave 1 (W1c). i8 not tested here: torch int8
sub wraps, oxtorch saturates — divergent by design; i8 is gated by the Rust test
(sub/mod.rs::parity, new vs legacy, which used correct SSE2).
"""


def test_sub_f32(parity):
    parity("sub", "f32", shapes=[(1024,), (33, 31)], tol=0.0)


def test_sub_f16(parity):
    parity("sub", "f16", shapes=[(1024,), (33, 31)])


def test_sub_bf16(parity):
    parity("sub", "bf16", shapes=[(1024,), (33, 31)])
