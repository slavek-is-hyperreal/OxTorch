//! SSE2 sigmoid: reuses the validated exp SSE2 core; 1/(1+exp(-x)).
//! Edge cases inherited from the exp core (no extra masks). See sigmoid_spec.md.
//!
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_sse2::exp4;

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sig4(x: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);
    let z = exp4(_mm_or_ps(x, _mm_set1_ps(-0.0))); // exp(-|x|), overflow-free
    let mask = _mm_cmplt_ps(x, _mm_setzero_ps());  // x < 0
    let num = _mm_or_ps(_mm_and_ps(mask, z), _mm_andnot_ps(mask, one));
    _mm_div_ps(num, _mm_add_ps(one, z))
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn sigmoid(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), sig4(_mm_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n4..n { out_buf[i] = super::sigmoid_f32_scalar::sigmoid_one(in_buf[i]); }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn sigmoid_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm_storeu_ps(ptr, sig4(_mm_loadu_ps(ptr)));
    }
    for x in buf[n4..].iter_mut() { *x = super::sigmoid_f32_scalar::sigmoid_one(*x); }
}
