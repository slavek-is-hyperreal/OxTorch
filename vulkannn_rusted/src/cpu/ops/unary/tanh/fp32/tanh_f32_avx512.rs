//! AVX-512 tanh — Cephes two-branch, k-mask select.
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_avx512::exp16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn tanh16(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let two = _mm512_set1_ps(2.0);
    let absmask = _mm512_set1_ps(f32::from_bits(0x7fff_ffff));
    let ax = _mm512_and_ps(x, absmask);

    let s = exp16(_mm512_add_ps(ax, ax));
    let big_mag = _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(s, one)));
    let signbits = _mm512_andnot_ps(absmask, x);
    let big = _mm512_or_ps(big_mag, signbits);

    let z = _mm512_mul_ps(x, x);
    let mut p = _mm512_set1_ps(-5.70498872745E-3);
    p = _mm512_add_ps(_mm512_mul_ps(p, z), _mm512_set1_ps(2.06390887954E-2));
    p = _mm512_add_ps(_mm512_mul_ps(p, z), _mm512_set1_ps(-5.37397155531E-2));
    p = _mm512_add_ps(_mm512_mul_ps(p, z), _mm512_set1_ps(1.33314422036E-1));
    p = _mm512_add_ps(_mm512_mul_ps(p, z), _mm512_set1_ps(-3.33332819422E-1));
    let small = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(p, z), x), x);

    let m = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(ax, _mm512_set1_ps(0.625));
    _mm512_mask_blend_ps(m, big, small)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn tanh(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) { _mm512_storeu_ps(out_buf.as_mut_ptr().add(i), tanh16(_mm512_loadu_ps(in_buf.as_ptr().add(i)))); }
    for i in n16..n { out_buf[i] = in_buf[i].tanh(); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn tanh_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) { let p = buf.as_mut_ptr().add(i); _mm512_storeu_ps(p, tanh16(_mm512_loadu_ps(p))); }
    for x in buf[n16..].iter_mut() { *x = x.tanh(); }
}
