//! SSE2 gelu (tanh-approx) — reuses tanh SSE2 core.
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::tanh::fp32::tanh_f32_sse2::tanh4;

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gelu4(x: __m128) -> __m128 {
    let k = _mm_set1_ps(0.7978845608); let c = _mm_set1_ps(0.044715);
    let half = _mm_set1_ps(0.5); let one = _mm_set1_ps(1.0);
    let x3 = _mm_mul_ps(_mm_mul_ps(x, x), x);
    let inner = _mm_mul_ps(k, _mm_add_ps(x, _mm_mul_ps(c, x3)));
    _mm_mul_ps(_mm_mul_ps(half, x), _mm_add_ps(one, tanh4(inner)))
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn gelu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { _mm_storeu_ps(out_buf.as_mut_ptr().add(i), gelu4(_mm_loadu_ps(in_buf.as_ptr().add(i)))); }
    for i in n4..n { out_buf[i] = super::gelu_f32_scalar::gelu_one(in_buf[i]); }
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn gelu_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { let p = buf.as_mut_ptr().add(i); _mm_storeu_ps(p, gelu4(_mm_loadu_ps(p))); }
    for x in buf[n4..].iter_mut() { *x = super::gelu_f32_scalar::gelu_one(*x); }
}
