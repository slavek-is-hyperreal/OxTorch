//! AVX-512 f32 sum — f64 accumulation (widen 8 f32 -> 8 f64 via _mm512_cvtps_pd).
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn sum(buf: &[f32]) -> f64 {
    let mut a0 = _mm512_setzero_pd();
    let mut a1 = _mm512_setzero_pd();
    let n16 = (buf.len() / 16) * 16;
    let mut i = 0;
    while i < n16 {
        let v0 = _mm256_loadu_ps(buf.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(buf.as_ptr().add(i + 8));
        a0 = _mm512_add_pd(a0, _mm512_cvtps_pd(v0));
        a1 = _mm512_add_pd(a1, _mm512_cvtps_pd(v1));
        i += 16;
    }
    let s = _mm512_add_pd(a0, a1);
    _mm512_reduce_add_pd(s) + buf[n16..].iter().map(|&x| x as f64).sum::<f64>()
}
