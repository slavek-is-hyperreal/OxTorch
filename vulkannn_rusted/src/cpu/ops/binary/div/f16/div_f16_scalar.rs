//! Baseline Scalar Implementation for F16 Divide (via f32 round-trip, guarded).
//! Transcribed from cpu_old/ops/binary/div/div_f16.rs. /0 -> 0.0 (legacy quirk).

#[inline(always)]
pub fn div(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    for i in 0..a.len() {
        let vb = b[i].to_f32();
        res[i] = half::f16::from_f32(if vb != 0.0 { a[i].to_f32() / vb } else { 0.0 });
    }
}
