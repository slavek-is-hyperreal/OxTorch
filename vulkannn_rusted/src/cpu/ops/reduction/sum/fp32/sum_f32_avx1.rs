//! AVX1 f32 sum — f64 ACCUMULATION (widen f32->f64 before adding).
//! Four __m256d accumulators (4 f64 each); f32 lanes are converted with
//! `_mm256_cvtps_pd` so no add ever happens in f32. See sum_f32_scalar for the
//! f64-accumulator policy (Rule 6 / Wave-3).
//!
//! BENCH: 4.5–7.7x vs the naive f64 scalar loop (i5-3450, `cargo bench -- sum_f32`,
//! 2026-07): 4K 7.7x, 64K 6.4x, 1M 4.5x. rustc does NOT auto-vectorise an f64 sum
//! (associativity), so the hand widen-accumulate wins big — a clear keeper.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn sum(buf: &[f32]) -> f64 {
    let mut a0 = _mm256_setzero_pd();
    let mut a1 = _mm256_setzero_pd();
    let mut a2 = _mm256_setzero_pd();
    let mut a3 = _mm256_setzero_pd();
    let n16 = (buf.len() / 16) * 16;
    let mut i = 0;
    while i < n16 {
        let v0 = _mm256_loadu_ps(buf.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(buf.as_ptr().add(i + 8));
        a0 = _mm256_add_pd(a0, _mm256_cvtps_pd(_mm256_castps256_ps128(v0)));
        a1 = _mm256_add_pd(a1, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0)));
        a2 = _mm256_add_pd(a2, _mm256_cvtps_pd(_mm256_castps256_ps128(v1)));
        a3 = _mm256_add_pd(a3, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v1)));
        i += 16;
    }
    // horizontal f64 sum of the 4 accumulators (16 f64 lanes)
    let s = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), s);
    let mut acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for &x in &buf[n16..] {
        acc += x as f64;
    }
    acc
}
