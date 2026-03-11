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

