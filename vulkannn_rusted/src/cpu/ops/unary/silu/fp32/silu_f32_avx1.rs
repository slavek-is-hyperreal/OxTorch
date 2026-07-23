//! AVX1 silu: x/(1+exp(-x)), reusing the exp AVX1 core.
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_avx1::exp8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn silu8(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(x, _mm256_add_ps(one, e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn silu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) { _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), silu8(_mm256_loadu_ps(in_buf.as_ptr().add(i)))); }
    for i in n8..n { out_buf[i] = super::silu_f32_scalar::silu_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn silu_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) { let p = buf.as_mut_ptr().add(i); _mm256_storeu_ps(p, silu8(_mm256_loadu_ps(p))); }
    for x in buf[n8..].iter_mut() { *x = super::silu_f32_scalar::silu_one(*x); }
}
