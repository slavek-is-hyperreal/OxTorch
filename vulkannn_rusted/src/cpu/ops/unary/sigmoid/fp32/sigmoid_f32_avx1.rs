//! AVX1 sigmoid: reuses the validated exp AVX1 core; 1/(1+exp(-x)).
//!
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_avx1::exp8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sig8(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let z = exp8(_mm256_or_ps(x, _mm256_set1_ps(-0.0))); // exp(-|x|)
    let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_setzero_ps());
    let num = _mm256_blendv_ps(one, z, mask);
    _mm256_div_ps(num, _mm256_add_ps(one, z))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn sigmoid(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), sig8(_mm256_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n8..n { out_buf[i] = super::sigmoid_f32_scalar::sigmoid_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn sigmoid_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, sig8(_mm256_loadu_ps(ptr)));
    }
    for x in buf[n8..].iter_mut() { *x = super::sigmoid_f32_scalar::sigmoid_one(*x); }
}
