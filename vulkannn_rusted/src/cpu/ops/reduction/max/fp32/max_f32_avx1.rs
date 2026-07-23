//! AVX1 f32 max (_mm256_max_ps). Ignores NaN (legacy, transcribed).
//! BENCH: PENDING (needs a unary/reduction bench harness). Memory-bound.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn max(buf: &[f32], initial: f32) -> f32 {
    let mut m = _mm256_set1_ps(initial); let n8 = (buf.len()/8)*8;
    for i in (0..n8).step_by(8) { m = _mm256_max_ps(m, _mm256_loadu_ps(buf.as_ptr().add(i))); }
    let mut t = [0f32;8]; _mm256_storeu_ps(t.as_mut_ptr(), m);
    let mut r = t.iter().fold(initial, |a,&b| a.max(b));
    for &x in &buf[n8..] { r = r.max(x); } r
}
