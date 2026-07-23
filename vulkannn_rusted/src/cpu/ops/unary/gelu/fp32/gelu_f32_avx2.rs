//! AVX2 gelu (tanh-approx) — reuses the validated tanh AVX1 core.
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::tanh::fp32::tanh_f32_avx2::tanh8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gelu8(x: __m256) -> __m256 {
    let k = _mm256_set1_ps(0.7978845608);
    let c = _mm256_set1_ps(0.044715);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    let inner = _mm256_mul_ps(k, _mm256_add_ps(x, _mm256_mul_ps(c, x3)));
    let t = tanh8(inner);
    _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, t))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gelu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), gelu8(_mm256_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n8..n { out_buf[i] = super::gelu_f32_scalar::gelu_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gelu_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, gelu8(_mm256_loadu_ps(ptr)));
    }
    for x in buf[n8..].iter_mut() { *x = super::gelu_f32_scalar::gelu_one(*x); }
}
