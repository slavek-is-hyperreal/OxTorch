//! SSE2 silu: x/(1+exp(-x)), reusing the validated exp SSE2 core.
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_sse2::exp4;

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn silu4(x: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);
    let e = exp4(_mm_sub_ps(_mm_setzero_ps(), x)); // exp(-x)
    _mm_div_ps(x, _mm_add_ps(one, e))
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn silu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), silu4(_mm_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n4..n { out_buf[i] = super::silu_f32_scalar::silu_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn silu_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm_storeu_ps(ptr, silu4(_mm_loadu_ps(ptr)));
    }
    for x in buf[n4..].iter_mut() { *x = super::silu_f32_scalar::silu_one(*x); }
}
