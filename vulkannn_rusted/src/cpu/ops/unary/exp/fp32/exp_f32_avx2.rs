//! AVX2 exp for FP32 — same Cephes math as the validated AVX1 kernel, but with
//! native 256-bit integer ops for the 2^n build (AVX2 has add/shift on __m256i).
//! Coeffs: docs/kernel_specs/exp_spec.md.
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn exp8(x: __m256) -> __m256 {
    let log2ef = _mm256_set1_ps(1.44269504088896341);
    let c1 = _mm256_set1_ps(0.693359375);
    let c2 = _mm256_set1_ps(-2.12194440e-4);
    let maxlogf = _mm256_set1_ps(88.72283905206835);
    let minlogf = _mm256_set1_ps(-103.278929903431851103);

    let xc = _mm256_min_ps(_mm256_max_ps(x, minlogf), maxlogf);
    let n = _mm256_cvtps_epi32(_mm256_mul_ps(log2ef, xc));
    let fn_ = _mm256_cvtepi32_ps(n);

    let mut g = _mm256_sub_ps(xc, _mm256_mul_ps(fn_, c1));
    g = _mm256_sub_ps(g, _mm256_mul_ps(fn_, c2));

    let mut p = _mm256_set1_ps(1.9875691500E-4);
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(1.3981999507E-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(8.3334519073E-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(4.1665795894E-2));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(1.6666665459E-1));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(5.0000001201E-1));

    let gg = _mm256_mul_ps(g, g);
    let mut eg = _mm256_add_ps(_mm256_mul_ps(p, gg), g);
    eg = _mm256_add_ps(eg, _mm256_set1_ps(1.0));

    // two-step ldexp (denormal-safe), native 256-bit int
    let bias = _mm256_set1_epi32(127);
    let n1 = _mm256_srai_epi32::<1>(n);
    let n2 = _mm256_sub_epi32(n, n1);
    let pow2a = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias)));
    let pow2b = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias)));
    let mut res = _mm256_mul_ps(_mm256_mul_ps(eg, pow2a), pow2b);

    let inf = _mm256_set1_ps(f32::INFINITY);
    res = _mm256_blendv_ps(res, inf, _mm256_cmp_ps::<_CMP_GT_OQ>(x, maxlogf));
    res = _mm256_blendv_ps(res, _mm256_setzero_ps(), _mm256_cmp_ps::<_CMP_LT_OQ>(x, minlogf));
    res = _mm256_blendv_ps(res, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
    res
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), exp8(_mm256_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n8..n {
        out_buf[i] = in_buf[i].exp();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn exp_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, exp8(_mm256_loadu_ps(ptr)));
    }
    for x in buf[n8..].iter_mut() {
        *x = x.exp();
    }
}
