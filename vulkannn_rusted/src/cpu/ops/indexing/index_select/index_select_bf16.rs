use half::bf16;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Wektoryzowany index_select dla BF16.
/// Architekturowo identyczny do F16, opiera się na kopiowaniu strumieniowym bloków 16-bit.
pub fn index_select_bf16(weight: &[bf16], indices: &[i32], out: &mut [bf16], feature_len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { index_select_bf16_avx512(weight, indices, out, feature_len) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { index_select_bf16_avx2(weight, indices, out, feature_len) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { index_select_bf16_avx1(weight, indices, out, feature_len) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return index_select_bf16_neon(weight, indices, out, feature_len);
    }

    index_select_bf16_swar(weight, indices, out, feature_len);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn index_select_bf16_avx512(weight: &[bf16], indices: &[i32], out: &mut [bf16], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let src_start = (idx as usize) * feature_len;
        let in_ptr = weight.as_ptr().add(src_start);
        
        _mm_prefetch(in_ptr as *const i8, _MM_HINT_T0);
        
        let mut f = 0;
        // 512-bit ZMM register hold 32 x 16-bit elements
        while f + 32 <= feature_len {
            let vec = _mm512_loadu_si512(in_ptr.add(f) as *const __m512i);
            _mm512_storeu_si512(out_ptr.add(f) as *mut __m512i, vec);
            f += 32;
        }
        
        let tail = feature_len - f;
        if tail > 0 {
            let mask = (1 << tail) - 1;
            let vec = _mm512_maskz_loadu_epi16(mask, in_ptr.add(f) as *const i16);
            _mm512_mask_storeu_epi16(out_ptr.add(f) as *mut i16, mask, vec);
        }
        out_ptr = out_ptr.add(feature_len);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn index_select_bf16_avx2(weight: &[bf16], indices: &[i32], out: &mut [bf16], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = weight.as_ptr().add((idx as usize) * feature_len);
        
        _mm_prefetch(in_ptr as *const i8, _MM_HINT_T0);
        
        let mut f = 0;
        // 256-bit YMM register holds 16 x 16-bit elements
        while f + 16 <= feature_len {
            let vec = _mm256_loadu_si256(in_ptr.add(f) as *const __m256i);
            _mm256_storeu_si256(out_ptr.add(f) as *mut __m256i, vec);
            f += 16;
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
unsafe fn index_select_bf16_avx1(weight: &[bf16], indices: &[i32], out: &mut [bf16], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = weight.as_ptr().add((idx as usize) * feature_len);
        
        _mm_prefetch(in_ptr as *const i8, _MM_HINT_T0);
        
        let mut f = 0;
        while f + 16 <= feature_len {
            let vec = _mm256_loadu_ps(in_ptr.add(f) as *const f32);
            _mm256_storeu_ps(out_ptr.add(f) as *mut f32, vec);
            f += 16;
        }
        while f < feature_len {
            *out_ptr.add(f) = *in_ptr.add(f);
            f += 1;
        }
        out_ptr = out_ptr.add(feature_len);
    }
}

#[cfg(target_arch = "aarch64")]
fn index_select_bf16_neon(weight: &[bf16], indices: &[i32], out: &mut [bf16], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = unsafe { weight.as_ptr().add((idx as usize) * feature_len) };
        let mut f = 0;
        while f + 8 <= feature_len {
            unsafe {
                let vec = vld1q_s16(in_ptr.add(f) as *const i16);
                vst1q_s16(out_ptr.add(f) as *mut i16, vec);
            }
            f += 8;
        }
        while f < feature_len {
            unsafe { *out_ptr.add(f) = *in_ptr.add(f) };
            f += 1;
        }
        unsafe { out_ptr = out_ptr.add(feature_len) };
    }
}

fn index_select_bf16_swar(weight: &[bf16], indices: &[i32], out: &mut [bf16], feature_len: usize) {
    let mut out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let in_ptr = unsafe { weight.as_ptr().add((idx as usize) * feature_len) };
        let mut f = 0;
        
        while f + 4 <= feature_len {
            unsafe {
                let chunk = std::ptr::read_unaligned(in_ptr.add(f) as *const u64);
                std::ptr::write_unaligned(out_ptr.add(f) as *mut u64, chunk);
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
