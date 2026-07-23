//! AVX-512 sigmoid: reuses the exp AVX-512 core; 1/(1+exp(-x)).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_avx512::exp16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn sig16(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let z = exp16(_mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(x), _mm512_castps_si512(_mm512_set1_ps(-0.0)))));
    let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_setzero_ps());
    let num = _mm512_mask_blend_ps(mask, one, z);
    _mm512_div_ps(num, _mm512_add_ps(one, z))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        _mm512_storeu_ps(out_buf.as_mut_ptr().add(i), sig16(_mm512_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n16..n { out_buf[i] = super::sigmoid_f32_scalar::sigmoid_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm512_storeu_ps(ptr, sig16(_mm512_loadu_ps(ptr)));
    }
    for x in buf[n16..].iter_mut() { *x = super::sigmoid_f32_scalar::sigmoid_one(*x); }
}
