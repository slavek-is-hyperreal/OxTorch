//! Baseline Scalar Implementation for F16 Multiply (via f32 round-trip).
//! Semantic reference — transcribed from cpu_old/ops/binary/mul/mul_f16.rs.

#[inline(always)]
pub fn mul(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    for i in 0..a.len() {
        res[i] = half::f16::from_f32(a[i].to_f32() * b[i].to_f32());
    }
}
