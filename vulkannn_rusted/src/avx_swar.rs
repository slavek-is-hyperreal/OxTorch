use half::{bf16, f16};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Convert a slice of f32 to bf16
pub fn convert_f32_to_bf16(src: &[f32], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { convert_f32_to_bf16_avx2(src, dst); return; }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { convert_f32_to_bf16_sse2(src, dst); return; }
        }
    }
    
    // Scalar fallback
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = bf16::from_f32(*s));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_f32_to_bf16_avx2(src: &[f32], dst: &mut [bf16]) {
    let chunks = src.chunks_exact(8);
    let rem = chunks.remainder();
    let mut dst_chunks = dst.chunks_exact_mut(8);
    
    // Round to nearest even bias: 0x7FFF + ((f32_bits >> 16) & 1)
    let bias_const = _mm256_set1_epi32(0x7FFF);
    let one_const = _mm256_set1_epi32(1);

    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| {
            unsafe {
                let vec = _mm256_loadu_ps(s.as_ptr());
                let vec_i = _mm256_castps_si256(vec);
                
                // Rounding
                let shifted_bits = _mm256_srli_epi32(vec_i, 16);
                let and_one = _mm256_and_si256(shifted_bits, one_const);
                let bias = _mm256_add_epi32(bias_const, and_one);
                let rounded = _mm256_add_epi32(vec_i, bias);
                let final_bits = _mm256_srli_epi32(rounded, 16);

                let low_half = _mm256_castsi256_si128(final_bits);
                let high_half = _mm256_extracti128_si256(final_bits, 1);
                
                let mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 13, 12, 9, 8, 5, 4, 1, 0);
                let p_low = _mm_shuffle_epi8(low_half, mask);
                let p_high = _mm_shuffle_epi8(high_half, mask);
                
                let mut res = [0u16; 8];
                _mm_storeu_si64(res.as_mut_ptr() as *mut u8, p_low);
                _mm_storeu_si64(res[4..].as_mut_ptr() as *mut u8, p_high);
                
                for i in 0..8 { d[i] = bf16::from_bits(res[i]); }
            }
        });
    
    let rem_start = (src.len() / 8) * 8;
    for (s, d) in src[rem_start..].iter().zip(dst[rem_start..].iter_mut()) {
        *d = bf16::from_f32(*s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_f32_to_bf16_sse2(src: &[f32], dst: &mut [bf16]) {
    let chunks = src.chunks_exact(4);
    let rem = chunks.remainder();
    let mut dst_chunks = dst.chunks_exact_mut(4);
    
    let bias_const = _mm_set1_epi32(0x7FFF);
    let one_const = _mm_set1_epi32(1);

    for (s, d) in chunks.zip(dst_chunks.by_ref()) {
        let vec = _mm_loadu_ps(s.as_ptr());
        let vec_i = _mm_castps_si128(vec);
        
        // Rounding
        let shifted_bits = _mm_srli_epi32(vec_i, 16);
        let and_one = _mm_and_si128(shifted_bits, one_const);
        let bias = _mm_add_epi32(bias_const, and_one);
        let rounded = _mm_add_epi32(vec_i, bias);
        let final_bits = _mm_srli_epi32(rounded, 16);
        
        // Pack into 64 bits (Low 16 bits of each 32-bit lane)
        #[cfg(target_feature = "ssse3")]
        {
            let mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 7, 6, 3, 2); // BUG in my mental mask, fixing:
            // Actually, just stores are simpler if we don't have a reliable mask here.
            // Let's use a safe store-based approach for SSE2.
        }
        
        let mut tmp = [0u16; 4];
        let bits_ptr = &final_bits as *const __m128i as *const u32;
        for i in 0..4 { tmp[i] = unsafe { *bits_ptr.add(i) } as u16; }
        for i in 0..4 { d[i] = bf16::from_bits(tmp[i]); }
    }
    
    for (s, d) in rem.iter().zip(dst_chunks.into_remainder().iter_mut()) {
        *d = bf16::from_f32(*s);
    }
}

/// Convert a slice of f32 to f16
pub fn convert_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            unsafe { convert_f32_to_f16_f16c(src, dst); return; }
        }
    }
    
    // Scalar fallback if no hardware F16C (Ivy Bridge)
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = f16::from_f32(*s));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn convert_f32_to_f16_f16c(src: &[f32], dst: &mut [f16]) {
    let chunks = src.chunks_exact(8);
    let rem = chunks.remainder();
    let mut dst_chunks = dst.chunks_exact_mut(8);
    
    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| {
            unsafe {
                let vec = _mm256_loadu_ps(s.as_ptr());
                let half_vec = _mm256_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, half_vec);
            }
        });
    
    let rem_start = (src.len() / 8) * 8;
    for (s, d) in src[rem_start..].iter().zip(dst[rem_start..].iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

/// Convert a slice of bf16 to f32
pub fn convert_bf16_to_f32(src: &[bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe { convert_bf16_to_f32_sse2(src, dst); return; }
        }
    }
    
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = s.to_f32());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_bf16_to_f32_sse2(src: &[bf16], dst: &mut [f32]) {
    let chunks = src.chunks_exact(4);
    let rem = chunks.remainder();
    let mut dst_chunks = dst.chunks_exact_mut(4);
    
    for (s, d) in chunks.zip(dst_chunks.by_ref()) {
        let mut tmp = [0u32; 4];
        for i in 0..4 { tmp[i] = (s[i].to_bits() as u32) << 16; }
        let vec = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
        let vec_f = _mm_castsi128_ps(vec);
        _mm_storeu_ps(d.as_mut_ptr(), vec_f);
    }
    
    for (s, d) in rem.iter().zip(dst_chunks.into_remainder().iter_mut()) {
        *d = s.to_f32();
    }
}

/// Convert a slice of f16 to f32
pub fn convert_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            unsafe { convert_f16_to_f32_f16c(src, dst); return; }
        }
    }
    
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = s.to_f32());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn convert_f16_to_f32_f16c(src: &[f16], dst: &mut [f32]) {
    let chunks = src.chunks_exact(8);
    let rem = chunks.remainder();
    let mut dst_chunks = dst.chunks_exact_mut(8);
    
    src.par_chunks_exact(8)
        .zip(dst.par_chunks_exact_mut(8))
        .for_each(|(s, d)| {
            unsafe {
                let vec = _mm_loadu_si128(s.as_ptr() as *const __m128i);
                let f32_vec = _mm256_cvtph_ps(vec);
                _mm256_storeu_ps(d.as_mut_ptr(), f32_vec);
            }
        });
    
    let rem_start = (src.len() / 8) * 8;
    for (s, d) in src[rem_start..].iter().zip(dst[rem_start..].iter_mut()) {
        *d = s.to_f32();
    }
}
