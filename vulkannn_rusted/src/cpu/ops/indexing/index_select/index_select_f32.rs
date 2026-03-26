#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Wektoryzowany index_select dla F32 (Embeddings Layer).
/// Szanuje struktury cache L1 dzięki _mm_prefetch (HINT_T0).
pub fn index_select_f32(weight: &[f32], indices: &[i32], out: &mut [f32], feature_len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { index_select_f32_avx512(weight, indices, out, feature_len) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { index_select_f32_avx2(weight, indices, out, feature_len) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { index_select_f32_avx1(weight, indices, out, feature_len) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // TODO: Neon
        return index_select_f32_neon(weight, indices, out, feature_len);
    }

    index_select_f32_scalar(weight, indices, out, feature_len);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn index_select_f32_avx512(weight: &[f32], indices: &[i32], out: &mut [f32], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let src_start = (idx as usize) * feature_len;
        let in_ptr = weight.as_ptr().add(src_start);
        
        // HPC PREFETCH: Ściągamy wiersze wagi bezpośrednio do L1 Cache
        _mm_prefetch(in_ptr as *const i8, _MM_HINT_T0);
        
        let mut f = 0;
        while f + 16 <= feature_len {
            let vec = _mm512_loadu_ps(in_ptr.add(f));
            _mm512_storeu_ps(out_ptr.add(f), vec);
            f += 16;
        }
        
        // AVX-512 Zero-masking tail handling bez użycia skalarnej pętli!
        let tail = feature_len - f;
        if tail > 0 {
            let mask = (1 << tail) - 1; // Generuj bitmaskę długości tail
            let vec = _mm512_maskz_loadu_ps(mask, in_ptr.add(f));
            _mm512_mask_storeu_ps(out_ptr.add(f), mask, vec);
        }
        out_ptr = out_ptr.add(feature_len);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn index_select_f32_avx2(weight: &[f32], indices: &[i32], out: &mut [f32], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = weight.as_ptr().add((idx as usize) * feature_len);
        
        _mm_prefetch(in_ptr as *const i8, _MM_HINT_T0);
        
        let mut f = 0;
        while f + 8 <= feature_len {
            let vec = _mm256_loadu_ps(in_ptr.add(f));
            _mm256_storeu_ps(out_ptr.add(f), vec);
            f += 8;
        }
        while f < feature_len {
            *out_ptr.add(f) = *in_ptr.add(f);
            f += 1;
        }
        out_ptr = out_ptr.add(feature_len);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn index_select_f32_avx1(weight: &[f32], indices: &[i32], out: &mut [f32], feature_len: usize) {
    // Te same instrukcje load/store co AVX2 dla f32
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = weight.as_ptr().add((idx as usize) * feature_len);
        
        _mm_prefetch(in_ptr as *const i8, _MM_HINT_T0);
        
        let mut f = 0;
        while f + 8 <= feature_len {
            let vec = _mm256_loadu_ps(in_ptr.add(f));
            _mm256_storeu_ps(out_ptr.add(f), vec);
            f += 8;
        }
        while f < feature_len {
            *out_ptr.add(f) = *in_ptr.add(f);
            f += 1;
        }
        out_ptr = out_ptr.add(feature_len);
    }
}

#[cfg(target_arch = "aarch64")]
fn index_select_f32_neon(weight: &[f32], indices: &[i32], out: &mut [f32], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = unsafe { weight.as_ptr().add((idx as usize) * feature_len) };
        let mut f = 0;
        while f + 4 <= feature_len {
            unsafe {
                let vec = vld1q_f32(in_ptr.add(f));
                vst1q_f32(out_ptr.add(f), vec);
            }
            f += 4;
        }
        while f < feature_len {
            unsafe { *out_ptr.add(f) = *in_ptr.add(f) };
            f += 1;
        }
        unsafe { out_ptr = out_ptr.add(feature_len) };
    }
}

fn index_select_f32_scalar(weight: &[f32], indices: &[i32], out: &mut [f32], feature_len: usize) {
    for (i, &idx) in indices.iter().enumerate() {
        let src_start = (idx as usize) * feature_len;
        let dst_start = i * feature_len;
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight.as_ptr().add(src_start),
                out.as_mut_ptr().add(dst_start),
                feature_len,
            );
        }
    }
}
