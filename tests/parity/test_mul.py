"""Parity: oxtorch mul kernels vs PyTorch (Wave 1).

The Rust half (cpu/ops/binary/mul/mod.rs::parity) proves mul matches the legacy
kernel bit-for-bit; this proves the whole stack matches torch. f32 is exact;
f16/bf16 use the dtype's default tolerance (oxtorch computes in f32, rounds once).
MANDATORY_LENS (SIMD-unfriendly lengths) and SPECIALS (±0/±inf/NaN/denormals)
are swept automatically by the fixture.
"""


def test_mul_f32(parity):
    parity("mul", "f32", shapes=[(1024,), (1, 1024), (33, 31)], tol=0.0)


def test_mul_f16(parity):
    parity("mul", "f16", shapes=[(1024,), (33, 31)])


def test_mul_bf16(parity):
    parity("mul", "bf16", shapes=[(1024,), (33, 31)])
