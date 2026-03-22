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
    // No vdivq_f32 in some NEON versions? Actually it is in AArch64.
    // Use reciprocal estimate for speed if needed, but let's try direct div first.
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
