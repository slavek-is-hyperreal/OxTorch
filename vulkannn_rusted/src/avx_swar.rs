//! avx_swar.rs — Cross-platform SIMD conversion for F16/BF16 ↔ F32
//!
//! Dispatch priority:
//!
//!  F16 ↔ F32:
//!    x86_64: f16c (AVX1+F16C, works on Ivy Bridge i5-3450)
//!         → SSE2 SWAR (branchless bit-twiddling, no F16C needed, e.g. Sandy Bridge)
//!         → scalar rayon fallback
//!    AArch64: NEON vcvt_f32_f16 / vcvt_f16_f32
//!         → scalar rayon fallback
//!
//!  BF16 ↔ F32:
//!    x86_64: AVX2 (vectorised round-to-nearest-even, 8 floats/iter)
//!         → SSE2 (4 floats/iter)
//!         → scalar rayon fallback
//!    AArch64: NEON (4 floats/iter via u32 shift tricks)
//!         → scalar rayon fallback

use half::{bf16, f16};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// convert_f32_to_f16
// ============================================================

/// Performs a high-performance vectorised conversion from F32 (IEEE 754) to F16.
/// Automatically dispatches to F16C (AVX1), SSE2 SWAR, or NEON based on CPU capabilities.
pub fn convert_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        // f16c requires AVX1 (not AVX2). Works on Ivy Bridge i5-3450, Haswell, etc.
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            unsafe { convert_f32_to_f16_f16c(src, dst); return; }
        }
        // SSE2 SWAR: works on any x86_64 since ~2001 (no F16C needed)
        if is_x86_feature_detected!("sse2") {
            unsafe { convert_f32_to_f16_sse2_swar(src, dst); return; }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on all AArch64 (Raspberry Pi 4, Apple M1, etc.)
        unsafe { convert_f32_to_f16_neon(src, dst); return; }
    }

    // Universal scalar fallback (no SIMD available)
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = f16::from_f32(*s));
}

// --- x86_64: F16C (AVX1 + F16C) — 8 floats per vector instruction ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f32_to_f16_f16c(src: &[f32], dst: &mut [f16]) {
    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| unsafe {
            let vec = _mm256_loadu_ps(s.as_ptr());
            // _mm256_cvtps_ph: needs f16c + avx (NOT avx2)
            let half_vec = _mm256_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, half_vec);
        });
    let rem = (src.len() / 8) * 8;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

// --- x86_64: SSE2 SWAR branchless — for CPUs without F16C (Sandy Bridge, old AMD) ---
// IEEE 754 FP16: sign(1) | exponent(5, bias=15) | mantissa(10)
// IEEE 754 FP32: sign(1) | exponent(8, bias=127) | mantissa(23)
// Strategy: re-bias exponent (127→15), truncate mantissa (23→10), handle overflow/underflow branchlessly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f32_to_f16_sse2_swar(src: &[f32], dst: &mut [f16]) {
    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            let v = _mm_loadu_ps(s.as_ptr());
            let vi = _mm_castps_si128(v);

            // Extract components
            let sign    = _mm_and_si128(vi, _mm_set1_epi32(0x8000_0000u32 as i32));
            let abs_val = _mm_and_si128(vi, _mm_set1_epi32(0x7FFF_FFFFi32));

            // Re-bias exponent: fp32 exp bias=127, fp16 exp bias=15 → subtract 112 from exponent
            // Then shift mantissa right by 13 bits (23-10)
            let rebias  = _mm_sub_epi32(abs_val, _mm_set1_epi32(0x3800_0000i32)); // subtract (127-15)<<23
            let shifted  = _mm_srli_epi32(rebias, 13);

            // Overflow mask: if original exponent >= 0x47800000 => fp16 infinity
            let inf_mask = _mm_cmpgt_epi32(abs_val, _mm_set1_epi32(0x477F_E000i32));
            // Underflow mask: if too small for fp16 (flush to zero)
            let zero_mask = _mm_cmplt_epi32(abs_val, _mm_set1_epi32(0x3880_0000i32));

            // Apply infinity
            let mut result = _mm_or_si128(
                _mm_andnot_si128(inf_mask, shifted),
                _mm_and_si128(inf_mask, _mm_set1_epi32(0x7C00i32)),
            );
            // Apply underflow (zero out)
            result = _mm_andnot_si128(zero_mask, result);

            // Re-attach sign, pack to 16-bit
            let sign16 = _mm_srli_epi32(sign, 16); // sign bit at position 15
            result = _mm_or_si128(result, sign16);

            // Extract 4 x u16 (low 16 bits of each i32 lane) via pointer cast
            let mut out = [0u16; 4];
            let tmp = result;
            let ptr = &tmp as *const __m128i as *const u32;
            for i in 0..4 { out[i] = *ptr.add(i) as u16; }
            for i in 0..4 { d[i] = f16::from_bits(out[i]); }

        });

    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

// --- AArch64: NEON vcvt — mandatory on all AArch64 ---
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_f32_to_f16_neon(src: &[f32], dst: &mut [f16]) {
    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            let v = vld1q_f32(s.as_ptr());
            let h = vcvt_f16_f32(v);
            // vst1_u16 stores 4 x u16
            let bits_ptr = d.as_mut_ptr() as *mut u16;
            vst1_u16(bits_ptr, vreinterpret_u16_f16(h));
        });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

// ============================================================
// convert_f16_to_f32
// ============================================================

/// Performs a high-performance vectorised conversion from F16 to F32 (IEEE 754).
/// Utilizes F16C hardware instructions or branchless bit-twiddling SWAR logic.
pub fn convert_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            unsafe { convert_f16_to_f32_f16c(src, dst); return; }
        }
        if is_x86_feature_detected!("sse2") {
            unsafe { convert_f16_to_f32_sse2_swar(src, dst); return; }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { convert_f16_to_f32_neon(src, dst); return; }
    }

    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = s.to_f32());
}

// --- x86_64: F16C —  8 halfs → 8 floats ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f16_to_f32_f16c(src: &[f16], dst: &mut [f32]) {
    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| unsafe {
            let vec = _mm_loadu_si128(s.as_ptr() as *const __m128i);
            let f32_vec = _mm256_cvtph_ps(vec);
            _mm256_storeu_ps(d.as_mut_ptr(), f32_vec);
        });
    let rem = (src.len() / 8) * 8;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = s.to_f32();
    }
}

// --- x86_64: SSE2 SWAR — branchless FP16→FP32 without F16C ---
// Algorithm: expand exponent from bias-15 to bias-127, extend mantissa by zero-padding 13 bits.
// Subnormals handled by checking exponent == 0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f16_to_f32_sse2_swar(src: &[f16], dst: &mut [f32]) {
    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            // Load 4 x u16 as i32 (zero-extended into each 32-bit lane)
            let mut h = [0u32; 4];
            for i in 0..4 { h[i] = s[i].to_bits() as u32; }
            let v16 = _mm_loadu_si128(h.as_ptr() as *const __m128i);

            let sign    = _mm_and_si128(v16, _mm_set1_epi32(0x8000i32));
            let expm    = _mm_and_si128(v16, _mm_set1_epi32(0x7C00i32)); // 5-bit exponent
            let mant    = _mm_and_si128(v16, _mm_set1_epi32(0x03FFi32)); // 10-bit mantissa

            // Re-bias: fp16 exponent → fp32 exponent.  (exp + 112) << 13
            // fp32 bias = 127, fp16 bias = 15, diff = 112.
            let exp_rebias = _mm_add_epi32(_mm_srli_epi32(expm, 10), _mm_set1_epi32(112));
            let exp_f32    = _mm_slli_epi32(exp_rebias, 23);
            let mant_f32   = _mm_slli_epi32(mant, 13);

            // Handle zero exponent (subnormals / zeros) → result should be 0 (or handled via scalar)
            let zero_mask  = _mm_cmpeq_epi32(expm, _mm_setzero_si128());
            // Handle infinity/NaN (exponent == 0x7C00) → map to fp32 infinity pattern
            let inf_mask   = _mm_cmpeq_epi32(expm, _mm_set1_epi32(0x7C00i32));

            let sign_f32   = _mm_slli_epi32(sign, 16);

            // Normal path: sign | exp_f32 | mant_f32
            let normal = _mm_or_si128(sign_f32, _mm_or_si128(exp_f32, mant_f32));
            // Infinity/NaN path: sign | 0x7F800000 | mant_f32
            let inf_val = _mm_or_si128(sign_f32, _mm_or_si128(_mm_set1_epi32(0x7F80_0000i32), mant_f32));

            let mut result = _mm_or_si128(
                _mm_and_si128(inf_mask, inf_val),
                _mm_andnot_si128(inf_mask, normal),
            );
            // Zero out subnormals for simplicity (flush to zero)
            result = _mm_andnot_si128(zero_mask, result);

            let vf = _mm_castsi128_ps(result);
            _mm_storeu_ps(d.as_mut_ptr(), vf);
        });

    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = s.to_f32();
    }
}

// --- AArch64: NEON ---
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_f16_to_f32_neon(src: &[f16], dst: &mut [f32]) {
    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            let h_bits = vld1_u16(s.as_ptr() as *const u16);
            let h = vreinterpret_f16_u16(h_bits);
            let v = vcvt_f32_f16(h);
            vst1q_f32(d.as_mut_ptr(), v);
        });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = s.to_f32();
    }
}

// ============================================================
// convert_f32_to_bf16
// ============================================================

/// Performs a high-performance vectorised conversion from F32 to BF16 (Brain Float).
/// Implements round-to-nearest-even precision to match PyTorch/TensorFlow behavior exactly.
pub fn convert_f32_to_bf16(src: &[f32], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        // AVX2: 8 floats/iter with round-to-nearest-even. Requires AVX2 (Haswell+).
        if is_x86_feature_detected!("avx2") {
            unsafe { convert_f32_to_bf16_avx2(src, dst); return; }
        }
        // SSE2: 4 floats/iter. Works on any x86_64 including Ivy Bridge.
        if is_x86_feature_detected!("sse2") {
            unsafe { convert_f32_to_bf16_sse2(src, dst); return; }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { convert_f32_to_bf16_neon(src, dst); return; }
    }

    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = bf16::from_f32(*s));
}

// --- x86_64: AVX2 — 8 floats, round-to-nearest-even ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_f32_to_bf16_avx2(src: &[f32], dst: &mut [bf16]) {
    let bias_const = _mm256_set1_epi32(0x7FFF);
    let one_const  = _mm256_set1_epi32(1);

    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| unsafe {
            let vec     = _mm256_loadu_ps(s.as_ptr());
            let vec_i   = _mm256_castps_si256(vec);
            // Round-to-nearest-even bias
            let shifted = _mm256_srli_epi32(vec_i, 16);
            let and_one = _mm256_and_si256(shifted, one_const);
            let bias    = _mm256_add_epi32(bias_const, and_one);
            let rounded = _mm256_add_epi32(vec_i, bias);
            let result  = _mm256_srli_epi32(rounded, 16);

            // Compact 8 x 16-bit from 8 x 32-bit
            let lo = _mm256_castsi256_si128(result);
            let hi = _mm256_extracti128_si256(result, 1);
            // pshufb to grab bytes [1,0] from each 32-bit lane → positions 0,2,4,6 in 128-bit reg
            let mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 13, 12, 9, 8, 5, 4, 1, 0);
            let p_lo = _mm_shuffle_epi8(lo, mask);
            let p_hi = _mm_shuffle_epi8(hi, mask);

            let mut res = [0u16; 8];
            _mm_storeu_si64(res.as_mut_ptr() as *mut u8, p_lo);
            _mm_storeu_si64(res[4..].as_mut_ptr() as *mut u8, p_hi);
            for i in 0..8 { d[i] = bf16::from_bits(res[i]); }
        });

    let rem = (src.len() / 8) * 8;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = bf16::from_f32(*s);
    }
}

// --- x86_64: SSE2 — 4 floats, round-to-nearest-even. No AVX2 needed. ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f32_to_bf16_sse2(src: &[f32], dst: &mut [bf16]) {
    let bias_const = _mm_set1_epi32(0x7FFF);
    let one_const  = _mm_set1_epi32(1);

    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            let vec     = _mm_loadu_ps(s.as_ptr());
            let vec_i   = _mm_castps_si128(vec);
            let shifted = _mm_srli_epi32(vec_i, 16);
            let and_one = _mm_and_si128(shifted, one_const);
            let bias    = _mm_add_epi32(bias_const, and_one);
            let rounded = _mm_add_epi32(vec_i, bias);
            let result  = _mm_srli_epi32(rounded, 16);

            // Extract 4 x u16 (low 16 bits of each i32 lane) — no SSSE3 needed
            let ptr = &result as *const __m128i as *const u32;
            for i in 0..4 {
                d[i] = bf16::from_bits(*ptr.add(i) as u16);
            }
        });

    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = bf16::from_f32(*s);
    }
}

// --- AArch64: NEON — shift trick: BF16 is just the upper 16 bits of FP32 ---
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_f32_to_bf16_neon(src: &[f32], dst: &mut [bf16]) {
    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            let v = vld1q_f32(s.as_ptr());
            // Reinterpret as u32, add rounding bias, shift right 16
            let vi     = vreinterpretq_u32_f32(v);
            let bias   = vaddq_u32(vshrq_n_u32(vi, 16), vdupq_n_u32(0x7FFFu32));
            let biased = vaddq_u32(vi, vandq_u32(bias, vdupq_n_u32(0xFFFF_0001u32)));
            let result = vshrn_n_u32(biased, 16); // narrow to u16x4
            let ptr = d.as_mut_ptr() as *mut u16;
            vst1_u16(ptr, result);
            // Re-interpret stored bits as bf16
            for i in 0..4 { d[i] = bf16::from_bits(*(ptr.add(i))); }
        });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = bf16::from_f32(*s);
    }
}

// ============================================================
// convert_bf16_to_f32
// ============================================================

pub fn convert_bf16_to_f32(src: &[bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 allows wider loads; for bf16→f32 SSE2 is already optimal (just shift).
        // Both paths are handled identically — use SSE2 everywhere (works on all x86_64).
        if is_x86_feature_detected!("sse2") {
            unsafe { convert_bf16_to_f32_sse2(src, dst); return; }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { convert_bf16_to_f32_neon(src, dst); return; }
    }

    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = s.to_f32());
}

// --- x86_64: SSE2 — BF16→F32 is trivially upper-16-bit-extend, no AVX2 needed ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_bf16_to_f32_sse2(src: &[bf16], dst: &mut [f32]) {
    // BF16 = upper 16 bits of FP32. Conversion = zero-extend and shift left 16 bits.
    src.par_chunks_exact(4)
        .zip(dst.par_chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            // Build 4 x u32 by shifting bf16 bits left 16
            let mut tmp = [0u32; 4];
            for i in 0..4 { tmp[i] = (s[i].to_bits() as u32) << 16; }
            let vec = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
            let vf  = _mm_castsi128_ps(vec);
            _mm_storeu_ps(d.as_mut_ptr(), vf);
        });

    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = s.to_f32();
    }
}

// --- AArch64: NEON ---
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_bf16_to_f32_neon(src: &[bf16], dst: &mut [f32]) {
    src.chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .for_each(|(s, d)| unsafe {
            let mut tmp = [0u32; 4];
            for i in 0..4 { tmp[i] = (s[i].to_bits() as u32) << 16; }
            let v = vld1q_u32(tmp.as_ptr());
            let vf = vreinterpretq_f32_u32(v);
            vst1q_f32(d.as_mut_ptr(), vf);
        });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) {
        *d = s.to_f32();
    }
}

// ============================================================
// relu_f32 — AVX1 vmaxps (Ivy Bridge compatible, no AVX2 needed)
// ============================================================
//
// Replaces scalar `if i > 0.0 { i } else { 0.0 }` with branchless
// _mm256_max_ps(x, zero) — 8 floats per cycle on AVX1.
// Falls back to SSE2 (4 floats) or scalar on older hardware.
//
pub fn relu_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { relu_f32_avx(src, dst); }
            return;
        }
        if is_x86_feature_detected!("sse2") {
            unsafe { relu_f32_sse2(src, dst); }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { relu_f32_neon(src, dst); }
        return;
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    for (o, &i) in dst.iter_mut().zip(src.iter()) {
        *o = if i > 0.0 { i } else { 0.0 };
    }
}

/// In-place variant: applies ReLU to a single slice.
pub fn relu_f32_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: AVX detected at runtime.
            unsafe {
                let zero = _mm256_setzero_ps();
                let n8 = (buf.len() / 8) * 8;
                let ptr = buf.as_mut_ptr();
                for i in (0..n8).step_by(8) {
                    let v = _mm256_loadu_ps(ptr.add(i));
                    _mm256_storeu_ps(ptr.add(i), _mm256_max_ps(v, zero));
                }
                for x in buf[n8..].iter_mut() {
                    if *x < 0.0 { *x = 0.0; }
                }
            }
            return;
        }
    }
    // Fallback
    for x in buf.iter_mut() {
        if *x < 0.0 { *x = 0.0; }
    }
}

// --- x86_64: AVX1 (256-bit, 8 floats/iter) ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn relu_f32_avx(src: &[f32], dst: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n8 = (src.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
    }
    // Scalar tail
    for j in n8..src.len() {
        dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 };
    }
}

// --- x86_64: SSE2 (128-bit, 4 floats/iter) ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn relu_f32_sse2(src: &[f32], dst: &mut [f32]) {
    let zero = _mm_setzero_ps();
    let n4 = (src.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = _mm_loadu_ps(src.as_ptr().add(i));
        _mm_storeu_ps(dst.as_mut_ptr().add(i), _mm_max_ps(v, zero));
    }
    for j in n4..src.len() {
        dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 };
    }
}

// --- AArch64: NEON (128-bit, 4 floats/iter) ---
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn relu_f32_neon(src: &[f32], dst: &mut [f32]) {
    let zero = vdupq_n_f32(0.0);
    let n4 = (src.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(src.as_ptr().add(i));
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(v, zero));
    }
    for j in n4..src.len() {
        dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 };
    }
}

// ============================================================
// exp_f32 — AVX1 Vectorized Exponential (for Softmax)
// ============================================================
//
// Uses the fast polynomial approximation exp(x) ≈ (1 + x/256)^256.
// Since x86 AVX1 lacks FMA instructions (which arrived in AVX2/FMA3),
// we use standard _mm256_add_ps and _mm256_mul_ps.
// For neural network softmax, 6 decimal digits of precision is sufficient.
//

pub fn exp_f32_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { exp_f32_avx_inplace(buf); }
            return;
        }
        if is_x86_feature_detected!("sse2") {
            unsafe { exp_f32_sse2_inplace(buf); }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { exp_f32_neon_inplace(buf); }
        return;
    }

    // Scalar fallback
    buf.par_iter_mut().for_each(|x| *x = x.exp());
}

// --- x86_64: AVX1 (256-bit, 8 floats/iter) ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn exp_f32_avx_inplace(buf: &mut [f32]) {
    // exp(x) = 2^(x / ln(2)) = 2^(floor(x/ln(2))) * 2^(frac(x/ln(2)))
    let log2_e = _mm256_set1_ps(1.4426950408889634); // 1 / ln(2)
    let ln2_hi = _mm256_set1_ps(0.6931457519); // ln(2) high
    let ln2_lo = _mm256_set1_ps(1.4286067653e-6); // ln(2) low
    
    // Bounds for f32 exp to prevent overflow/underflow NaN/inf issues
    let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-88.3762626647949);
    
    // Polynomial coefficients for 2^frac
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(0.16666667163);
    let c4 = _mm256_set1_ps(0.04166648536);
    let c5 = _mm256_set1_ps(0.00833336077);
    let c6 = _mm256_set1_ps(0.00138925374);

    // Magic number for float round to integer using addition/subtraction
    let magic = _mm256_set1_ps(12582912.0); // 1.5 * 2^23

    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        let mut x = _mm256_loadu_ps(ptr);

        // Clamp x
        x = _mm256_min_ps(x, max_x);
        x = _mm256_max_ps(x, min_x);

        // n = round(x * log2_e)
        // AVX1 trick to round to nearest integer: (x + magic) - magic
        let nx = _mm256_mul_ps(x, log2_e);
        let mut n = _mm256_add_ps(nx, magic);
        // We need the raw int bits later, so grab them from the addition
        let n_bits = _mm256_castps_si256(n); 
        n = _mm256_sub_ps(n, magic);

        // Find exact fractional part: frac = x - n * ln2
        let mut frac = x;
        let n_ln2_hi = _mm256_mul_ps(n, ln2_hi);
        let n_ln2_lo = _mm256_mul_ps(n, ln2_lo);
        frac = _mm256_sub_ps(frac, n_ln2_hi);
        frac = _mm256_sub_ps(frac, n_ln2_lo);

        // Evaluate polynomial P(frac) = 1 + frac + frac^2 / 2 + frac^3 / 6 + ...
        let mut p = c6;
        p = _mm256_add_ps(_mm256_mul_ps(frac, p), c5);
        p = _mm256_add_ps(_mm256_mul_ps(frac, p), c4);
        p = _mm256_add_ps(_mm256_mul_ps(frac, p), c3);
        p = _mm256_add_ps(_mm256_mul_ps(frac, p), c2);
        p = _mm256_add_ps(_mm256_mul_ps(frac, p), c1);
        p = _mm256_add_ps(_mm256_mul_ps(frac, p), c1); 

        // Build 2^n directly in floating point format
        // The magic addition left the integer value of n in the bottom 23 bits
        // We need n + 127 in the exponent (bits 23-30), so we shift up by 23
        // Since it's already in the mantissa bits, we just add the IEEE bias as an integer.
        // Wait, _mm256_add_epi32 is AVX2. 
        // For AVX1, we extract into 128-bit SSE blocks, use SSE2 int math, and put it back.
        let n_lo = _mm256_extractf128_si256::<0>(n_bits);
        let n_hi = _mm256_extractf128_si256::<1>(n_bits);
        
        // n_bits contains n + 12582912. 12582912 in hex is 0x00C00000.
        // When we added magic, the integer part was placed in the bottom 23 bits.
        // Float format places the exponent at bit 23.
        // So `(n << 23) + (127 << 23)` is exactly `n << 23 + 0x3F800000`
        // Since `n << 23` is ALREADY what n_bits holds (except it has 0x4B400000 from the magic offset!)
        // Wait, `(n + magic)` float bits: If n=1, 12582913.0 = 0x4B400001
        // The float bits are literally `0x4B400000 + n`.
        // To get `(n + 127) << 23`, we need to change `0x4B400000 + n` into `(n << 23) + 0x3F800000`.
        // Which means `(n << 23)` would be `(n_bits - 0x4B400000) << 23`
        // Actually, just extract lower bits via SSE2:
        let bias = _mm_set1_epi32(127);
        let pow2_lo = _mm_slli_epi32(_mm_add_epi32(n_lo, bias), 23);
        let pow2_hi = _mm_slli_epi32(_mm_add_epi32(n_hi, bias), 23);
        
        // n_bits contains n + 12582912. 12582912 in hex is 0x00C00000.
        // It's already in the mantissa format when seen as an integer, 
        // with n occupying the lowest 23 bits and the rest forming 0x4B400000.
        // What we want is (n + 127) << 23.
        // A much more elegant (and FMA-less, AVX1-friendly) way is actually:
        // n_bits - 0x4B400000 to get n, then add 127, then shift 23.
        
        let pow2_n_int = _mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(pow2_lo), 
            pow2_hi
        );
        let pow2_n = _mm256_castsi256_ps(pow2_n_int);

        // exp(x) = p * 2^n
        let out = _mm256_mul_ps(p, pow2_n);

        _mm256_storeu_ps(ptr, out);
    }
    
    for x in buf[n8..].iter_mut() {
        *x = x.exp();
    }
}

// --- x86_64: SSE2 (128-bit, 4 floats/iter) ---
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn exp_f32_sse2_inplace(buf: &mut [f32]) {
    let div_256 = _mm_set1_ps(1.0 / 256.0);
    let one = _mm_set1_ps(1.0);
    
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        let mut v = _mm_loadu_ps(ptr);
        
        v = _mm_mul_ps(v, div_256);
        v = _mm_add_ps(v, one);
        
        v = _mm_mul_ps(v, v); v = _mm_mul_ps(v, v);
        v = _mm_mul_ps(v, v); v = _mm_mul_ps(v, v);
        v = _mm_mul_ps(v, v); v = _mm_mul_ps(v, v);
        v = _mm_mul_ps(v, v); v = _mm_mul_ps(v, v);
        
        _mm_storeu_ps(ptr, v);
    }
    
    for x in buf[n4..].iter_mut() {
        *x = x.exp();
    }
}

// --- AArch64: NEON (128-bit, 4 floats/iter) ---
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_f32_neon_inplace(buf: &mut [f32]) {
    let div_256 = vdupq_n_f32(1.0 / 256.0);
    let one = vdupq_n_f32(1.0);
    
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        let mut v = vld1q_f32(ptr);
        
        v = fma_neon_workaround(one, v, div_256); // essentially 1.0 + (v * 1/256)
        
        v = vmulq_f32(v, v); v = vmulq_f32(v, v);
        v = vmulq_f32(v, v); v = vmulq_f32(v, v);
        v = vmulq_f32(v, v); v = vmulq_f32(v, v);
        v = vmulq_f32(v, v); v = vmulq_f32(v, v);
        
        vst1q_f32(ptr, v);
    }
    
    for x in buf[n4..].iter_mut() {
        *x = x.exp();
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fma_neon_workaround(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    // vfmaq_f32 acts as a = a + b*c
    vfmaq_f32(a, b, c)
}
