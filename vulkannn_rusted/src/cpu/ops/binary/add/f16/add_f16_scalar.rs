//! Baseline Scalar Implementation for F16 Add (via f32 round-trip).
//! Transcribed from cpu_old/ops/binary/add/add_f16.rs.

#[inline(always)]
pub fn add(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    for i in 0..a.len() {
        res[i] = half::f16::from_f32(a[i].to_f32() + b[i].to_f32());
    }
}
