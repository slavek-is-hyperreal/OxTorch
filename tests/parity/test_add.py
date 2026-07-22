"""Parity: oxtorch add kernels vs PyTorch.

f32/bf16 were migrated in Wave 0, f16 in Wave 1 (W1c). i8 is intentionally NOT
tested here: torch's int8 add WRAPS on overflow while oxtorch SATURATES, so they
diverge by design; i8 correctness is gated by the Rust test (add/mod.rs::parity,
new vs the scalar saturating reference).
"""


def test_add_f32(parity):
    parity("add", "f32", shapes=[(1024,), (33, 31)], tol=0.0)


def test_add_f16(parity):
    parity("add", "f16", shapes=[(1024,), (33, 31)])


def test_add_bf16(parity):
    parity("add", "bf16", shapes=[(1024,), (33, 31)])
