//! F16 max — f16c widen + scalar. NaN ignored (legacy).
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
pub fn max(buf: &[half::f16], initial: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { max_f16c(buf, initial) };
        }
    }
    buf.iter().fold(initial, |a, &b| a.max(b.to_f32()))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn max_f16c(buf: &[half::f16], initial: f32) -> f32 {
    let mut m = _mm256_set1_ps(initial); let n8 = (buf.len()/8)*8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_cvtph_ps(_mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i));
        m = _mm256_max_ps(m, v);
    }
    let mut t=[0f32;8]; _mm256_storeu_ps(t.as_mut_ptr(), m);
    let mut r = t.iter().fold(initial, |a,&b| a.max(b));
    for &x in &buf[n8..] { r = r.max(x.to_f32()); } r
}
