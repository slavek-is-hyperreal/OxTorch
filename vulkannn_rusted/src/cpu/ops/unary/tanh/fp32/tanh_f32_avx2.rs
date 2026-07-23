//! AVX2 tanh for FP32 — Cephes tanhf two-branch (small-x poly + large-x via the
//! exp core). Coeffs: docs/kernel_specs/tanh_spec.md.
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_avx2::exp8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn tanh8(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let sign = _mm256_set1_ps(-0.0);
    let ax = _mm256_andnot_ps(sign, x); // |x|

    // large-x branch: copysign(1 - 2/(exp(2|x|)+1), x). exp overflow -> saturates.
    let s = exp8(_mm256_add_ps(ax, ax));
    let big_mag = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(s, one)));
    let big = _mm256_or_ps(big_mag, _mm256_and_ps(x, sign)); // copysign (big_mag>=0)

    // small-x branch: ((((P0*z+P1)*z+P2)*z+P3)*z+P4)*z*x + x, z=x*x.
    let z = _mm256_mul_ps(x, x);
    let mut p = _mm256_set1_ps(-5.70498872745E-3);
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(2.06390887954E-2));
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(-5.37397155531E-2));
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(1.33314422036E-1));
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(-3.33332819422E-1));
    let small = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(p, z), x), x);

    // select ax < 0.625
    let m = _mm256_cmp_ps::<_CMP_LT_OQ>(ax, _mm256_set1_ps(0.625));
    _mm256_blendv_ps(big, small, m)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn tanh(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), tanh8(_mm256_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n8..n { out_buf[i] = in_buf[i].tanh(); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn tanh_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, tanh8(_mm256_loadu_ps(ptr)));
    }
    for x in buf[n8..].iter_mut() { *x = x.tanh(); }
}
