//! AVX1 ReLU for FP32 (`_mm256_max_ps` with zero).
//! Transcribed from cpu_old/ops/unary/relu/relu_f32.rs (legacy avx kernel).
//!
//! BENCH: PENDING (needs a unary bench harness). Memory-bound (§8).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn relu(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(in_buf.as_ptr().add(i));
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
    }
    for i in n8..n {
        out_buf[i] = in_buf[i].max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn relu_inplace(buf: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, _mm256_max_ps(_mm256_loadu_ps(ptr), zero));
    }
    for x in buf[n8..].iter_mut() {
        *x = x.max(0.0);
    }
}
