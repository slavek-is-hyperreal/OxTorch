//! SSE2 tanh — Cephes two-branch, and/andnot/or select (no blendv).
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_sse2::exp4;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn tanh4(x: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);
    let two = _mm_set1_ps(2.0);
    let sign = _mm_set1_ps(-0.0);
    let ax = _mm_andnot_ps(sign, x);

    let s = exp4(_mm_add_ps(ax, ax));
    let big_mag = _mm_sub_ps(one, _mm_div_ps(two, _mm_add_ps(s, one)));
    let big = _mm_or_ps(big_mag, _mm_and_ps(x, sign));

    let z = _mm_mul_ps(x, x);
    let mut p = _mm_set1_ps(-5.70498872745E-3);
    p = _mm_add_ps(_mm_mul_ps(p, z), _mm_set1_ps(2.06390887954E-2));
    p = _mm_add_ps(_mm_mul_ps(p, z), _mm_set1_ps(-5.37397155531E-2));
    p = _mm_add_ps(_mm_mul_ps(p, z), _mm_set1_ps(1.33314422036E-1));
    p = _mm_add_ps(_mm_mul_ps(p, z), _mm_set1_ps(-3.33332819422E-1));
    let small = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(p, z), x), x);

    let m = _mm_cmplt_ps(ax, _mm_set1_ps(0.625));
    _mm_or_ps(_mm_and_ps(m, small), _mm_andnot_ps(m, big))
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn tanh(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), tanh4(_mm_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n4..n { out_buf[i] = in_buf[i].tanh(); }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn tanh_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm_storeu_ps(ptr, tanh4(_mm_loadu_ps(ptr)));
    }
    for x in buf[n4..].iter_mut() { *x = x.tanh(); }
}
