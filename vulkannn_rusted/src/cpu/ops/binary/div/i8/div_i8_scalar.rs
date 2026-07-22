//! Baseline Scalar Implementation for I8 Divide (saturating, guarded).
//! Transcribed VERBATIM from cpu_old/ops/binary/div/div_i8.rs. /0 -> 0 (legacy).
//! Legacy has NO SIMD i8-divide kernel, so none is fabricated (Rule 1).

#[inline(always)]
pub fn div(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = if b[i] != 0 { a[i].saturating_div(b[i]) } else { 0 };
    }
}
