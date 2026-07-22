//! Baseline Scalar Implementation for I8 Add (saturating).
//! Transcribed from cpu_old/ops/binary/add/add_i8.rs.

#[inline(always)]
pub fn add(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = a[i].saturating_add(b[i]);
    }
}
