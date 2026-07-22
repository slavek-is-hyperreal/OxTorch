//! I8 DIV — Tier II serial dispatcher. Scalar-only: legacy provides no SIMD
//! i8-divide kernel and integer SIMD division does not exist, so none is
//! fabricated (Rule 1). Saturating, /0 -> 0.

pub mod div_i8_scalar;

pub fn div_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    div_i8_scalar::div(a, b, res);
}
