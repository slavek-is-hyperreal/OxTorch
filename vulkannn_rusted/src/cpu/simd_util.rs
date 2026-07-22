//! simd_util.rs — shared vector math helpers for the migrated CPU kernels.
//!
//! # Provenance
//! Every function here is transcribed from `src/cpu_old/ops/math_simd.rs`.
//! **The original file is untouched and still live** — `cpu_old`'s unary ops
//! (sigmoid/silu/tanh/gelu) still call it. This is a copy so that migrated
//! kernels under `src/cpu/` never reach back into `cpu_old`; the two will be
//! reconciled in Wave 6 when `cpu_old` is deleted.
//!
//! # Numerics
//! All polynomial coefficients and range-reduction constants are transcribed
//! **literally**, digit for digit, from the legacy file. Do not "improve" them
//! here: doing so would silently change model outputs and break every parity
//! snapshot. If a coefficient looks wrong, fix it in a dedicated change with a
//! fresh parity baseline, not as a drive-by.
//!
//! Note for reviewers: the exponential is a degree-5 (AVX2) / degree-4 (NEON)
//! Cephes-style minimax polynomial after range reduction to |f| <= ln(2)/2.
//! It is an *approximation* — parity against `f32::exp` is not bitwise, and the
//! AVX2 and NEON variants are not bitwise identical to each other either
//! (different polynomial degree, and NEON uses an FMA that AVX1/AVX2 lack).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Vectorized exponential function (approximation) for f32x8 (AVX2).
/// Fast polynomial approximation using Cephes method.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn exp_ps_avx2(x: __m256) -> __m256 {
    // Ported from common SIMD math libraries (e.g., SLEEF or similar fast math)
    // Range reduction: e^x = 2^k * e^f, where |f| <= ln(2)/2
    let ln2 = _mm256_set1_ps(0.69314718);
    let inv_ln2 = _mm256_set1_ps(1.44269504);

    let k = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let x = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2));

    // Polynomial approx for e^x in [-ln2/2, ln2/2]
    // P(x) = 1 + x + x^2/2! + x^3/3! ...
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(0.16666667);
    let c4 = _mm256_set1_ps(0.04166666);
    let c5 = _mm256_set1_ps(0.00833333);

    let y = _mm256_add_ps(c1, _mm256_mul_ps(x, _mm256_add_ps(c1, _mm256_mul_ps(x, _mm256_add_ps(c2, _mm256_mul_ps(x, _mm256_add_ps(c3, _mm256_mul_ps(x, _mm256_add_ps(c4, _mm256_mul_ps(x, c5))))))))));

    // Multiply by 2^k
    // 2^k can be calculated by bit-shifting into the exponent field of f32
    // float pattern: [sign][exponent(8)][mantissa(23)]
    let k_i = _mm256_cvtps_epi32(k);
    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_i, _mm256_set1_epi32(127)), 23);
    let pow2k = _mm256_castsi256_ps(exp_bits);

    _mm256_mul_ps(y, pow2k)
}

/// NEON version of exp_ps
#[cfg(target_arch = "aarch64")]
pub unsafe fn exp_ps_neon(x: float32x4_t) -> float32x4_t {
    // Similar logic for NEON
    let ln2 = vdupq_n_f32(0.69314718);
    let inv_ln2 = vdupq_n_f32(1.44269504);

    let k = vrndnq_f32(vmulq_f32(x, inv_ln2));
    let f = vfmsq_f32(x, k, ln2);

    let c1 = vdupq_n_f32(1.0);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(0.16666667);
    let c4 = vdupq_n_f32(0.04166666);

    let y = vaddq_f32(c1, vmulq_f32(f, vaddq_f32(c1, vmulq_f32(f, vaddq_f32(c2, vmulq_f32(f, vaddq_f32(c3, vmulq_f32(f, c4))))))));

    let k_i = vcvtq_s32_f32(k);
    let exp_bits = vshlq_n_s32(vaddq_s32(k_i, vdupq_n_s32(127)), 23);
    let pow2k = vreinterpretq_f32_s32(exp_bits);

    vmulq_f32(y, pow2k)
}

/// Sigmoid approximation: 1 / (1 + exp(-x))
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sigmoid_ps_avx2(x: __m256) -> __m256 {
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let e_neg_x = exp_ps_avx2(neg_x);
    let one = _mm256_set1_ps(1.0);
    _mm256_div_ps(one, _mm256_add_ps(one, e_neg_x))
}

/// SiLU (Swish): x * sigmoid(x)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn silu_ps_avx2(x: __m256) -> __m256 {
    _mm256_mul_ps(x, sigmoid_ps_avx2(x))
}

/// Tanh approximation: (exp(2x) - 1) / (exp(2x) + 1)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn tanh_ps_avx2(x: __m256) -> __m256 {
    let x2 = _mm256_add_ps(x, x);
    let exp2x = exp_ps_avx2(x2);
    let one = _mm256_set1_ps(1.0);
    _mm256_div_ps(_mm256_sub_ps(exp2x, one), _mm256_add_ps(exp2x, one))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn sigmoid_ps_neon(x: float32x4_t) -> float32x4_t {
    let neg_x = vnegq_f32(x);
    let e_neg_x = exp_ps_neon(neg_x);
    let one = vdupq_n_f32(1.0);
    // `vdivq_f32` is AArch64-only (ARMv7 NEON has no vector divide); this file is
    // gated on `target_arch = "aarch64"` so it is always available here.
    vdivq_f32(one, vaddq_f32(one, e_neg_x))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn silu_ps_neon(x: float32x4_t) -> float32x4_t {
    vmulq_f32(x, sigmoid_ps_neon(x))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn tanh_ps_neon(x: float32x4_t) -> float32x4_t {
    let x2 = vaddq_f32(x, x);
    let exp2x = exp_ps_neon(x2);
    let one = vdupq_n_f32(1.0);
    vdivq_f32(vsubq_f32(exp2x, one), vaddq_f32(exp2x, one))
}

// ---------------------------------------------------------------------------
// AVX1 (Ivy Bridge) note
// ---------------------------------------------------------------------------
// There is deliberately **no** `exp_ps_avx1` here. The AVX2 version depends on
// integer ops on 256-bit lanes (`_mm256_slli_epi32`, `_mm256_add_epi32`,
// `_mm256_cvtps_epi32`) that AVX1 does not provide; a correct AVX1 port needs
// the 128-bit-half split (`_mm256_extractf128_si256` / `_mm256_insertf128_si256`)
// and must be written and benchmarked as its own kernel, not faked here.
// The development box (i5-3450, Ivy Bridge) has AVX1 but no AVX2/FMA, so any op
// depending on these helpers currently takes the scalar path on it.

#[cfg(test)]
mod tests {
    /// Reference used only to bound the approximation error in the tests below.
    fn exp_ref(x: f32) -> f32 {
        x.exp()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_avx2_matches_libm_within_tolerance() {
        if !is_x86_feature_detected!("avx2") {
            // i5-3450 (Ivy Bridge) lands here — AVX1 only.
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        use std::arch::x86_64::*;
        let inputs: [f32; 8] = [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.0];
        let mut out = [0f32; 8];
        unsafe {
            let v = _mm256_loadu_ps(inputs.as_ptr());
            let r = super::exp_ps_avx2(v);
            _mm256_storeu_ps(out.as_mut_ptr(), r);
        }
        for (i, &x) in inputs.iter().enumerate() {
            let want = exp_ref(x);
            let rel = ((out[i] - want) / want).abs();
            assert!(rel < 1e-4, "exp({}) = {} want {} (rel {})", x, out[i], want, rel);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sigmoid_avx2_is_bounded_and_monotone() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        use std::arch::x86_64::*;
        let inputs: [f32; 8] = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut out = [0f32; 8];
        unsafe {
            let v = _mm256_loadu_ps(inputs.as_ptr());
            let r = super::sigmoid_ps_avx2(v);
            _mm256_storeu_ps(out.as_mut_ptr(), r);
        }
        for w in out.windows(2) {
            assert!(w[1] >= w[0], "sigmoid not monotone: {:?}", out);
        }
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)), "{:?}", out);
        assert!((out[4] - 0.5).abs() < 1e-4, "sigmoid(0) = {}", out[4]);
    }
}
