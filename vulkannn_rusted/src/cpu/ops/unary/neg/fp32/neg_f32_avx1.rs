//! AVX1 negation for FP32 (`0 - x`). Transcribed from cpu_old neg_f32 (legacy avx).
//!
//! BENCH: PENDING (needs a unary bench harness). Memory-bound (§8).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn neg(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(in_buf.as_ptr().add(i));
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_sub_ps(zero, v));
    }
    for i in n8..n {
        out_buf[i] = -in_buf[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn neg_inplace(buf: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, _mm256_sub_ps(zero, _mm256_loadu_ps(ptr)));
    }
    for x in buf[n8..].iter_mut() {
        *x = -*x;
    }
}
