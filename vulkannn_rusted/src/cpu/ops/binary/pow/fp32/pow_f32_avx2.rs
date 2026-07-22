//! AVX2 square kernel for the FP32 Pow exponent==2.0 fast path.
//! Same op as AVX1 at the avx2 feature level (elementwise square).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn square(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(in_buf.as_ptr().add(i));
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_mul_ps(v, v));
    }
    for i in n8..n {
        out_buf[i] = in_buf[i] * in_buf[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn square_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(v, v));
    }
    for x in buf[n8..].iter_mut() {
        *x = *x * *x;
    }
}
