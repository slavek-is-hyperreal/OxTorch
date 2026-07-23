//! AVX-512 exp for FP32 — same Cephes math, 512-bit, k-mask edge selects.
//! Coeffs: docs/kernel_specs/exp_spec.md.
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn exp16(x: __m512) -> __m512 {
    let log2ef = _mm512_set1_ps(1.44269504088896341);
    let c1 = _mm512_set1_ps(0.693359375);
    let c2 = _mm512_set1_ps(-2.12194440e-4);
    let maxlogf = _mm512_set1_ps(88.72283905206835);
    let minlogf = _mm512_set1_ps(-103.278929903431851103);

    let xc = _mm512_min_ps(_mm512_max_ps(x, minlogf), maxlogf);
    let n = _mm512_cvtps_epi32(_mm512_mul_ps(log2ef, xc));
    let fn_ = _mm512_cvtepi32_ps(n);

    let mut g = _mm512_sub_ps(xc, _mm512_mul_ps(fn_, c1));
    g = _mm512_sub_ps(g, _mm512_mul_ps(fn_, c2));

    let mut p = _mm512_set1_ps(1.9875691500E-4);
    p = _mm512_add_ps(_mm512_mul_ps(p, g), _mm512_set1_ps(1.3981999507E-3));
    p = _mm512_add_ps(_mm512_mul_ps(p, g), _mm512_set1_ps(8.3334519073E-3));
    p = _mm512_add_ps(_mm512_mul_ps(p, g), _mm512_set1_ps(4.1665795894E-2));
    p = _mm512_add_ps(_mm512_mul_ps(p, g), _mm512_set1_ps(1.6666665459E-1));
    p = _mm512_add_ps(_mm512_mul_ps(p, g), _mm512_set1_ps(5.0000001201E-1));

    let gg = _mm512_mul_ps(g, g);
    let mut eg = _mm512_add_ps(_mm512_mul_ps(p, gg), g);
    eg = _mm512_add_ps(eg, _mm512_set1_ps(1.0));

    let bias = _mm512_set1_epi32(127);
    let n1 = _mm512_srai_epi32::<1>(n);
    let n2 = _mm512_sub_epi32(n, n1);
    let pow2a = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(n1, bias)));
    let pow2b = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(n2, bias)));
    let mut res = _mm512_mul_ps(_mm512_mul_ps(eg, pow2a), pow2b);

    let inf = _mm512_set1_ps(f32::INFINITY);
    let ov = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, maxlogf);
    let un = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, minlogf);
    let nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(x, x);
    res = _mm512_mask_blend_ps(ov, res, inf);
    res = _mm512_mask_blend_ps(un, res, _mm512_setzero_ps());
    res = _mm512_mask_blend_ps(nan, res, x);
    res
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        _mm512_storeu_ps(out_buf.as_mut_ptr().add(i), exp16(_mm512_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n16..n {
        out_buf[i] = in_buf[i].exp();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm512_storeu_ps(ptr, exp16(_mm512_loadu_ps(ptr)));
    }
    for x in buf[n16..].iter_mut() {
        *x = x.exp();
    }
}
