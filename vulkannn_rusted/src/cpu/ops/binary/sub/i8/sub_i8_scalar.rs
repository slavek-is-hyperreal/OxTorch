//! Baseline Scalar Implementation for I8 Sub (saturating).
//! Transcribed from cpu_old/ops/binary/sub/sub_i8.rs.

#[inline(always)]
pub fn sub(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = a[i].saturating_sub(b[i]);
    }
}
