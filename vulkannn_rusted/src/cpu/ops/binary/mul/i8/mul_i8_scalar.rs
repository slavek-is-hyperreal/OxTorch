//! Baseline Scalar Implementation for I8 Multiply (saturating).
//! Semantic reference — transcribed from cpu_old/ops/binary/mul/mul_i8.rs.

#[inline(always)]
pub fn mul(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = a[i].saturating_mul(b[i]);
    }
}
