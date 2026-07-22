//! F16C ReLU for F16 (hardware f16<->f32 convert + AVX max).
//! Transcribed from cpu_old/ops/unary/relu/relu_f16.rs (legacy f16c kernel).
//!
//! BENCH: PENDING (needs an f16 bench harness — deferred within Wave 2).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn relu(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    let zero = _mm256_setzero_ps();
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_in = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
        let f_res = _mm256_max_ps(_mm256_cvtph_ps(h_in), zero);
        _mm_storeu_si128(
            out_buf.as_mut_ptr().add(i) as *mut __m128i,
            _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(f_res),
        );
    }
    for i in n8..n {
        out_buf[i] = half::f16::from_f32(in_buf[i].to_f32().max(0.0));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn relu_inplace(buf: &mut [half::f16]) {
    let zero = _mm256_setzero_ps();
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i) as *mut __m128i;
        let f_res = _mm256_max_ps(_mm256_cvtph_ps(_mm_loadu_si128(ptr)), zero);
        _mm_storeu_si128(ptr, _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(f_res));
    }
    for x in buf[n8..].iter_mut() {
        *x = half::f16::from_f32(x.to_f32().max(0.0));
    }
}
