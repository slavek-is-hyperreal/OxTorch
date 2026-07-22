//! Baseline Scalar Implementation for FP32 Divide.
//! Semantic reference — transcribed from cpu_old/ops/binary/div/div_f32.rs.
//!
//! LEGACY QUIRK (Rule 6 — preserved verbatim, NOT fixed here): scalar guards
//! `b == 0` and returns 0.0, whereas the SIMD kernels use a raw hardware divide
//! (±inf/NaN on /0). Legacy is internally inconsistent this way — the SIMD body
//! yields inf on /0 while its scalar tail yields 0.0. This is the documented
//! oxtorch-vs-torch divergence; changing it is a post-migration decision.

#[inline(always)]
pub fn div(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, res.len());
    for i in 0..n {
        res[i] = if b[i] != 0.0 { a[i] / b[i] } else { 0.0 };
    }
}
