//! SSE2 f32 sum — f64 accumulation (widen 2 f32->2 f64 per cvt). no-AVX tier.
//! BENCH: PENDING — measurable here; expected between scalar and avx1.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn sum(buf: &[f32]) -> f64 {
    let mut a0 = _mm_setzero_pd();
    let mut a1 = _mm_setzero_pd();
    let n8 = (buf.len() / 8) * 8;
    let mut i = 0;
    while i < n8 {
        let v0 = _mm_loadu_ps(buf.as_ptr().add(i));
        let v1 = _mm_loadu_ps(buf.as_ptr().add(i + 4));
        a0 = _mm_add_pd(a0, _mm_cvtps_pd(v0));                 // low 2 f32
        a1 = _mm_add_pd(a1, _mm_cvtps_pd(_mm_movehl_ps(v0, v0))); // high 2 f32
        a0 = _mm_add_pd(a0, _mm_cvtps_pd(v1));
        a1 = _mm_add_pd(a1, _mm_cvtps_pd(_mm_movehl_ps(v1, v1)));
        i += 8;
    }
    let s = _mm_add_pd(a0, a1);
    let mut tmp = [0.0f64; 2];
    _mm_storeu_pd(tmp.as_mut_ptr(), s);
    let mut acc = tmp[0] + tmp[1];
    for &x in &buf[n8..] { acc += x as f64; }
    acc
}
