use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Per-token symmetric quantization (absmax).
/// Returns Int8 tensor and F32 scales (1 per token).
pub fn quantize_per_token_absmax_f32(_m: usize, k: usize, src: &[f32], dst: &mut [i8], scales: &mut [f32]) {
    dst.par_chunks_mut(k).enumerate().zip(scales.par_iter_mut()).for_each(|((i, dst_row), scale_out)| {
        let src_row = &src[i * k .. (i + 1) * k];
        
        let mut max_abs = 1e-7f32;
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let mut max_v = _mm256_set1_ps(1e-7f32);
                    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
                    
                    for chunk in src_row.chunks_exact(8) {
                        let v = _mm256_loadu_ps(chunk.as_ptr());
                        let abs_v = _mm256_and_ps(v, abs_mask);
                        max_v = _mm256_max_ps(max_v, abs_v);
                    }
                    
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
                    for &v in &tmp { if v > max_abs { max_abs = v; } }
                    
                    // Remainder
                    let rem = (src_row.len() / 8) * 8;
                    for &v in &src_row[rem..] {
                        let abs = v.abs();
                        if abs > max_abs { max_abs = abs; }
                    }
                }
            } else {
                for &v in src_row {
                    let abs = v.abs();
                    if abs > max_abs { max_abs = abs; }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for &v in src_row {
                let abs = v.abs();
                if abs > max_abs { max_abs = abs; }
            }
        }

        let scale = 127.0 / max_abs;
        *scale_out = max_abs / 127.0; // Store the dequantization scale
        
        for j in 0..k {
            dst_row[j] = (src_row[j] * scale).round().clamp(-128.0, 127.0) as i8;
        }
    });
}

pub fn quantize_per_token_absmax_bf16(_m: usize, k: usize, src: &[half::bf16], dst: &mut [i8], scales: &mut [f32]) {
    dst.par_chunks_mut(k).enumerate().zip(scales.par_iter_mut()).for_each(|((i, dst_row), scale_out)| {
        let src_row = &src[i * k .. (i + 1) * k];
        let mut max_abs = 1e-7f32;
        
        for &v in src_row {
            let abs = v.to_f32().abs();
            if abs > max_abs { max_abs = abs; }
        }

        let scale = 127.0 / max_abs;
        *scale_out = max_abs / 127.0;
        
        for j in 0..k {
            dst_row[j] = (src_row[j].to_f32() * scale).round().clamp(-128.0, 127.0) as i8;
        }
    });
}

pub fn quantize_per_token_absmax_f16(_m: usize, k: usize, src: &[half::f16], dst: &mut [i8], scales: &mut [f32]) {
    dst.par_chunks_mut(k).enumerate().zip(scales.par_iter_mut()).for_each(|((i, dst_row), scale_out)| {
        let src_row = &src[i * k .. (i + 1) * k];
        let mut max_abs = 1e-7f32;
        
        for &v in src_row {
            let abs = v.to_f32().abs();
            if abs > max_abs { max_abs = abs; }
        }

        let scale = 127.0 / max_abs;
        *scale_out = max_abs / 127.0;
        
        for j in 0..k {
            dst_row[j] = (src_row[j].to_f32() * scale).round().clamp(-128.0, 127.0) as i8;
        }
    });
}
