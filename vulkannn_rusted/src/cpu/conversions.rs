use half::{f16, bf16};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn convert_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            unsafe { convert_f32_to_f16_f16c(src, dst); return; }
        }
        if is_x86_feature_detected!("sse2") {
            unsafe { convert_f32_to_f16_sse2_swar(src, dst); return; }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { convert_f32_to_f16_neon(src, dst); return; }
    }
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = f16::from_f32(*s));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f32_to_f16_f16c(src: &[f32], dst: &mut [f16]) {
    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| unsafe {
            let vec = _mm256_loadu_ps(s.as_ptr());
            let half_vec = _mm256_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, half_vec);
        });
    let rem = (src.len() / 8) * 8;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = f16::from_f32(*s); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f32_to_f16_sse2_swar(src: &[f32], dst: &mut [f16]) {
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let v = _mm_loadu_ps(s.as_ptr());
        let vi = _mm_castps_si128(v);
        let sign = _mm_and_si128(vi, _mm_set1_epi32(0x8000_0000u32 as i32));
        let abs_val = _mm_and_si128(vi, _mm_set1_epi32(0x7FFF_FFFFi32));
        let rebias = _mm_sub_epi32(abs_val, _mm_set1_epi32(0x3800_0000i32));
        let shifted = _mm_srli_epi32(rebias, 13);
        let inf_mask = _mm_cmpgt_epi32(abs_val, _mm_set1_epi32(0x477F_E000i32));
        let zero_mask = _mm_cmplt_epi32(abs_val, _mm_set1_epi32(0x3880_0000i32));
        let mut result = _mm_or_si128(_mm_andnot_si128(inf_mask, shifted), _mm_and_si128(inf_mask, _mm_set1_epi32(0x7C00i32)));
        result = _mm_andnot_si128(zero_mask, result);
        let sign16 = _mm_srli_epi32(sign, 16);
        result = _mm_or_si128(result, sign16);
        let mut out = [0u16; 4];
        let ptr = &result as *const __m128i as *const u32;
        for i in 0..4 { out[i] = *ptr.add(i) as u16; }
        for i in 0..4 { d[i] = f16::from_bits(out[i]); }
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = f16::from_f32(*s); }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_f32_to_f16_neon(src: &[f32], dst: &mut [f16]) {
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let v = vld1q_f32(s.as_ptr());
        let h = vcvt_f16_f32(v);
        let bits_ptr = d.as_mut_ptr() as *mut u16;
        vst1_u16(bits_ptr, vreinterpret_u16_f16(h));
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = f16::from_f32(*s); }
}

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f16_to_f32_f16c(src: &[f16], dst: &mut [f32]) {
    src.par_chunks_exact(8).zip(dst.par_chunks_exact_mut(8)).for_each(|(s, d)| unsafe {
        let vec = _mm_loadu_si128(s.as_ptr() as *const __m128i);
        let f32_vec = _mm256_cvtph_ps(vec);
        _mm256_storeu_ps(d.as_mut_ptr(), f32_vec);
    });
    let rem = (src.len() / 8) * 8;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = s.to_f32(); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f16_to_f32_sse2_swar(src: &[f16], dst: &mut [f32]) {
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let mut h = [0u32; 4];
        for i in 0..4 { h[i] = s[i].to_bits() as u32; }
        let v16 = _mm_loadu_si128(h.as_ptr() as *const __m128i);
        let sign = _mm_and_si128(v16, _mm_set1_epi32(0x8000i32));
        let expm = _mm_and_si128(v16, _mm_set1_epi32(0x7C00i32));
        let mant = _mm_and_si128(v16, _mm_set1_epi32(0x03FFi32));
        let exp_rebias = _mm_add_epi32(_mm_srli_epi32(expm, 10), _mm_set1_epi32(112));
        let exp_f32    = _mm_slli_epi32(exp_rebias, 23);
        let mant_f32   = _mm_slli_epi32(mant, 13);
        let zero_mask  = _mm_cmpeq_epi32(expm, _mm_setzero_si128());
        let inf_mask   = _mm_cmpeq_epi32(expm, _mm_set1_epi32(0x7C00i32));
        let sign_f32   = _mm_slli_epi32(sign, 16);
        let normal = _mm_or_si128(sign_f32, _mm_or_si128(exp_f32, mant_f32));
        let inf_val = _mm_or_si128(sign_f32, _mm_or_si128(_mm_set1_epi32(0x7F80_0000i32), mant_f32));
        let mut result = _mm_or_si128(_mm_and_si128(inf_mask, inf_val), _mm_andnot_si128(inf_mask, normal));
        result = _mm_andnot_si128(zero_mask, result);
        _mm_storeu_ps(d.as_mut_ptr(), _mm_castsi128_ps(result));
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = s.to_f32(); }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_f16_to_f32_neon(src: &[f16], dst: &mut [f32]) {
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let h_bits = vld1_u16(s.as_ptr() as *const u16);
        let h = vreinterpret_f16_u16(h_bits);
        vst1q_f32(d.as_mut_ptr(), vcvt_f32_f16(h));
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = s.to_f32(); }
}

pub fn convert_f32_to_bf16(src: &[f32], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") { unsafe { convert_f32_to_bf16_avx2(src, dst); return; } }
        if is_x86_feature_detected!("sse2") { unsafe { convert_f32_to_bf16_sse2(src, dst); return; } }
    }
    #[cfg(target_arch = "aarch64")] { unsafe { convert_f32_to_bf16_neon(src, dst); return; } }
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = bf16::from_f32(*s));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_f32_to_bf16_avx2(src: &[f32], dst: &mut [bf16]) {
    let bias_const = _mm256_set1_epi32(0x7FFF);
    let one_const = _mm256_set1_epi32(1);
    src.par_chunks_exact(8).zip(dst.par_chunks_exact_mut(8)).for_each(|(s, d)| unsafe {
        let vec = _mm256_loadu_ps(s.as_ptr());
        let vec_i = _mm256_castps_si256(vec);
        let shifted = _mm256_srli_epi32(vec_i, 16);
        let and_one = _mm256_and_si256(shifted, one_const);
        let bias = _mm256_add_epi32(bias_const, and_one);
        let rounded = _mm256_add_epi32(vec_i, bias);
        let result = _mm256_srli_epi32(rounded, 16);
        let lo = _mm256_castsi256_si128(result);
        let hi = _mm256_extracti128_si256(result, 1);
        let mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 13, 12, 9, 8, 5, 4, 1, 0);
        let p_lo = _mm_shuffle_epi8(lo, mask);
        let p_hi = _mm_shuffle_epi8(hi, mask);
        let mut res = [0u16; 8];
        _mm_storeu_si64(res.as_mut_ptr() as *mut u8, p_lo);
        _mm_storeu_si64(res[4..].as_mut_ptr() as *mut u8, p_hi);
        for i in 0..8 { d[i] = bf16::from_bits(res[i]); }
    });
    let rem = (src.len() / 8) * 8;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = bf16::from_f32(*s); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f32_to_bf16_sse2(src: &[f32], dst: &mut [bf16]) {
    let bias_const = _mm_set1_epi32(0x7FFF);
    let one_const = _mm_set1_epi32(1);
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let vec = _mm_loadu_ps(s.as_ptr());
        let vec_i = _mm_castps_si128(vec);
        let shifted = _mm_srli_epi32(vec_i, 16);
        let and_one = _mm_and_si128(shifted, one_const);
        let bias = _mm_add_epi32(bias_const, and_one);
        let rounded = _mm_add_epi32(vec_i, bias);
        let result = _mm_srli_epi32(rounded, 16);
        let ptr = &result as *const __m128i as *const u32;
        for i in 0..4 { d[i] = bf16::from_bits(*ptr.add(i) as u16); }
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = bf16::from_f32(*s); }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_f32_to_bf16_neon(src: &[f32], dst: &mut [bf16]) {
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let v = vld1q_f32(s.as_ptr());
        let vi = vreinterpretq_u32_f32(v);
        let bias = vaddq_u32(vshrq_n_u32(vi, 16), vdupq_n_u32(0x7FFFu32));
        let biased = vaddq_u32(vi, vandq_u32(bias, vdupq_n_u32(0xFFFF_0001u32)));
        let result = vshrn_n_u32(biased, 16);
        let ptr = d.as_mut_ptr() as *mut u16;
        vst1_u16(ptr, result);
        for i in 0..4 { d[i] = bf16::from_bits(*(ptr.add(i))); }
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = bf16::from_f32(*s); }
}

pub fn convert_bf16_to_f32(src: &[bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") { unsafe { convert_bf16_to_f32_sse2(src, dst); return; } }
    }
    #[cfg(target_arch = "aarch64")] { unsafe { convert_bf16_to_f32_neon(src, dst); return; } }
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = s.to_f32());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_bf16_to_f32_sse2(src: &[bf16], dst: &mut [f32]) {
    src.par_chunks_exact(4).zip(dst.par_chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let mut tmp = [0u32; 4];
        for i in 0..4 { tmp[i] = (s[i].to_bits() as u32) << 16; }
        _mm_storeu_ps(d.as_mut_ptr(), _mm_castsi128_ps(_mm_loadu_si128(tmp.as_ptr() as *const __m128i)));
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = s.to_f32(); }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_bf16_to_f32_neon(src: &[bf16], dst: &mut [f32]) {
    src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).for_each(|(s, d)| unsafe {
        let mut tmp = [0u32; 4];
        for i in 0..4 { tmp[i] = (s[i].to_bits() as u32) << 16; }
        vst1q_f32(d.as_mut_ptr(), vreinterpretq_f32_u32(vld1q_u32(tmp.as_ptr())));
    });
    let rem = (src.len() / 4) * 4;
    for (s, d) in src[rem..].iter().zip(dst[rem..].iter_mut()) { *d = s.to_f32(); }
}
