//! SSE2 (no-AVX tier) exp for FP32 — Cephes expf polynomial + edge masks.
//! Same math as the validated AVX1 kernel at 128-bit width. SSE2 has no blendv,
//! so edge selects use and/andnot/or. Coeffs: docs/kernel_specs/exp_spec.md.
//!
//! BENCH: PENDING (needs a unary bench harness). Compute-bound.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn exp4(x: __m128) -> __m128 {
    let log2ef = _mm_set1_ps(1.44269504088896341);
    let c1 = _mm_set1_ps(0.693359375);
    let c2 = _mm_set1_ps(-2.12194440e-4);
    let maxlogf = _mm_set1_ps(88.72283905206835);
    let minlogf = _mm_set1_ps(-103.278929903431851103);

    let xc = _mm_min_ps(_mm_max_ps(x, minlogf), maxlogf);
    let n = _mm_cvtps_epi32(_mm_mul_ps(log2ef, xc));
    let fn_ = _mm_cvtepi32_ps(n);

    let mut g = _mm_sub_ps(xc, _mm_mul_ps(fn_, c1));
    g = _mm_sub_ps(g, _mm_mul_ps(fn_, c2));

    let mut p = _mm_set1_ps(1.9875691500E-4);
    p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(1.3981999507E-3));
    p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(8.3334519073E-3));
    p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(4.1665795894E-2));
    p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(1.6666665459E-1));
    p = _mm_add_ps(_mm_mul_ps(p, g), _mm_set1_ps(5.0000001201E-1));

    let gg = _mm_mul_ps(g, g);
    let mut eg = _mm_add_ps(_mm_mul_ps(p, gg), g);
    eg = _mm_add_ps(eg, _mm_set1_ps(1.0));

    // two-step ldexp (denormal-safe)
    let bias = _mm_set1_epi32(127);
    let n1 = _mm_srai_epi32::<1>(n);
    let n2 = _mm_sub_epi32(n, n1);
    let pow2a = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n1, bias)));
    let pow2b = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n2, bias)));
    let res = _mm_mul_ps(_mm_mul_ps(eg, pow2a), pow2b);

    // Edge selects without blendv: sel(m,a,b) = (m&a)|(~m&b).
    let inf = _mm_set1_ps(f32::INFINITY);
    let ov = _mm_cmpgt_ps(x, maxlogf);       // x > MAXLOGF (also +inf)  -> +inf
    let un = _mm_cmplt_ps(x, minlogf);       // x < MINLOGF (also -inf)  -> 0
    let nan = _mm_cmpunord_ps(x, x);         // NaN -> NaN (pass x)
    let mut r = _mm_or_ps(_mm_and_ps(ov, inf), _mm_andnot_ps(ov, res));
    r = _mm_andnot_ps(un, r);                // & ~un  (0 where underflow)
    r = _mm_or_ps(_mm_and_ps(nan, x), _mm_andnot_ps(nan, r));
    r
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), exp4(_mm_loadu_ps(in_buf.as_ptr().add(i))));
    }
    for i in n4..n {
        out_buf[i] = in_buf[i].exp();
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn exp_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm_storeu_ps(ptr, exp4(_mm_loadu_ps(ptr)));
    }
    for x in buf[n4..].iter_mut() {
        *x = x.exp();
    }
}
