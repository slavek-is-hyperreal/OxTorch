//! AVX1 exp for FP32 — Cephes expf polynomial + edge-case masks.
//! Coefficients: docs/kernel_specs/exp_spec.md (Cephes single/expf.c + constf.c).
//! AVX1 has no 256-bit integer ops (that is AVX2), so the `2^n` scale is built
//! from two 128-bit halves.
//!
//! BENCH: PENDING (needs a unary bench harness). Compute-bound — the matrix
//! earns its cost here; measure vs scalar on this box once the harness lands.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn exp8(x: __m256) -> __m256 {
    let log2ef = _mm256_set1_ps(1.44269504088896341);
    let c1 = _mm256_set1_ps(0.693359375);
    let c2 = _mm256_set1_ps(-2.12194440e-4);
    let maxlogf = _mm256_set1_ps(88.72283905206835);
    let minlogf = _mm256_set1_ps(-103.278929903431851103);

    // Clamp into the polynomial's valid range; the ends are fixed by masks below.
    let xc = _mm256_min_ps(_mm256_max_ps(x, minlogf), maxlogf);

    // n = round_to_nearest(log2ef * xc); fn_ = (float)n
    let n = _mm256_cvtps_epi32(_mm256_mul_ps(log2ef, xc));
    let fn_ = _mm256_cvtepi32_ps(n);

    // g = xc - fn*C1 - fn*C2  (Cody-Waite split of ln2)
    let mut g = _mm256_sub_ps(xc, _mm256_mul_ps(fn_, c1));
    g = _mm256_sub_ps(g, _mm256_mul_ps(fn_, c2));

    // Horner: p = ((((P0*g+P1)*g+P2)*g+P3)*g+P4)*g+P5
    let mut p = _mm256_set1_ps(1.9875691500E-4);
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(1.3981999507E-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(8.3334519073E-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(4.1665795894E-2));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(1.6666665459E-1));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(5.0000001201E-1));

    // e^g = p*g^2 + g + 1
    let gg = _mm256_mul_ps(g, g);
    let mut eg = _mm256_add_ps(_mm256_mul_ps(p, gg), g);
    eg = _mm256_add_ps(eg, _mm256_set1_ps(1.0));

    // Scale by 2^n via a two-step ldexp so denormals (n < -126) come out right:
    // 2^n = 2^n1 * 2^n2, n1 = n>>1, n2 = n - n1, each in [-127,127] for our n
    // range, so both bias-tricks stay in the normal exponent field and the two
    // IEEE multiplies produce the correct denormal / underflow-to-0.
    // AVX1 has no 256-bit int shift/add, so build each pow2 per 128-bit half.
    let bias = _mm_set1_epi32(127);
    let pow2 = |k: __m128i| -> __m128 { _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(k, bias))) };
    let n_lo = _mm256_castsi256_si128(n);
    let n_hi = _mm256_extractf128_si256::<1>(n);
    let build = |half: __m128i| -> (__m128, __m128) {
        let n1 = _mm_srai_epi32::<1>(half);           // n>>1 (arithmetic)
        let n2 = _mm_sub_epi32(half, n1);
        (pow2(n1), pow2(n2))
    };
    let (p1_lo, p2_lo) = build(n_lo);
    let (p1_hi, p2_hi) = build(n_hi);
    let pow2a = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(p1_lo), p1_hi);
    let pow2b = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(p2_lo), p2_hi);
    let mut res = _mm256_mul_ps(_mm256_mul_ps(eg, pow2a), pow2b);

    // Edge cases (exact): overflow/+inf -> +inf; underflow/-inf -> 0; NaN -> NaN.
    let inf = _mm256_set1_ps(f32::INFINITY);
    res = _mm256_blendv_ps(res, inf, _mm256_cmp_ps::<_CMP_GT_OQ>(x, maxlogf));
    res = _mm256_blendv_ps(res, _mm256_setzero_ps(), _mm256_cmp_ps::<_CMP_LT_OQ>(x, minlogf));
    res = _mm256_blendv_ps(res, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
    res
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(in_buf.as_ptr().add(i));
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), exp8(v));
    }
    for i in n8..n {
        out_buf[i] = in_buf[i].exp();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
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
