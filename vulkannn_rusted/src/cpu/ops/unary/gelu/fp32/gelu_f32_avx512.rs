//! AVX-512 gelu (tanh-approx) — reuses tanh AVX-512 core.
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::tanh::fp32::tanh_f32_avx512::tanh16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gelu16(x: __m512) -> __m512 {
    let k = _mm512_set1_ps(0.7978845608); let c = _mm512_set1_ps(0.044715);
    let half = _mm512_set1_ps(0.5); let one = _mm512_set1_ps(1.0);
    let x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
    let inner = _mm512_mul_ps(k, _mm512_add_ps(x, _mm512_mul_ps(c, x3)));
    _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, tanh16(inner)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) { _mm512_storeu_ps(out_buf.as_mut_ptr().add(i), gelu16(_mm512_loadu_ps(in_buf.as_ptr().add(i)))); }
    for i in n16..n { out_buf[i] = super::gelu_f32_scalar::gelu_one(in_buf[i]); }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) { let p = buf.as_mut_ptr().add(i); _mm512_storeu_ps(p, gelu16(_mm512_loadu_ps(p))); }
    for x in buf[n16..].iter_mut() { *x = super::gelu_f32_scalar::gelu_one(*x); }
}
