//! AVX-512 silu: x/(1+exp(-x)), reusing the exp AVX-512 core.
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_avx512::exp16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn silu16(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let e = exp16(_mm512_sub_ps(_mm512_setzero_ps(), x));
    _mm512_div_ps(x, _mm512_add_ps(one, e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) { _mm512_storeu_ps(out_buf.as_mut_ptr().add(i), silu16(_mm512_loadu_ps(in_buf.as_ptr().add(i)))); }
    for i in n16..n { out_buf[i] = super::silu_f32_scalar::silu_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) { let p = buf.as_mut_ptr().add(i); _mm512_storeu_ps(p, silu16(_mm512_loadu_ps(p))); }
    for x in buf[n16..].iter_mut() { *x = super::silu_f32_scalar::silu_one(*x); }
}
