#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise addition for BF16 tensors with multi-architecture support.
pub fn add_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    const PAR_THRESHOLD: usize = 256_000;
    if a.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            add_bf16_serial(ac, bc, rc);
        });
    } else {
        add_bf16_serial(a, b, res);
    }
}

fn add_bf16_serial(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") {
            return unsafe { add_bf16_avx(a, b, res) };
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return add_bf16_neon(a, b, res);
    }

    add_bf16_scalar(a, b, res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_bf16_avx(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n8 = (a.len() / 8) * 8;
    let zero = _mm_setzero_si128();
    for i in (0..n8).step_by(8) {
        let ba = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let bb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        
        // Upcast BF16 to F32
        let va = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_unpacklo_epi16(zero, ba)), _mm_unpackhi_epi16(zero, ba)));
        let vb = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_unpacklo_epi16(zero, bb)), _mm_unpackhi_epi16(zero, bb)));
        
        let vr = _mm256_add_ps(va, vb);
        
        // Downcast F32 to BF16 (truncate)
        let res_si = _mm256_castps_si256(vr);
        let h_lo = _mm_srli_epi32(_mm256_castsi256_si128(res_si), 16);
        let h_hi = _mm_srli_epi32(_mm256_extractf128_si256::<1>(res_si), 16);
        _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, _mm_packus_epi32(h_lo, h_hi));
    }
    add_bf16_scalar(&a[n8..], &b[n8..], &mut res[n8..]);
}

#[cfg(target_arch = "aarch64")]
fn add_bf16_neon(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    use std::arch::aarch64::*;
    let n4 = (a.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        unsafe {
            // BF16 to F32: move high 16 bits to high 16 bits of f32 results in upcast (if we zero low 16)
            let ba = vld1_u16(a.as_ptr().add(i) as *const u16);
            let bb = vld1_u16(b.as_ptr().add(i) as *const u16);
            
            // Manual upcast: (u16 << 16) as f32
            let va = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(ba), 16));
            let vb = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bb), 16));
            
            let vr = vaddq_f32(va, vb);
            
            // Downcast: (f32 >> 16) as u16
            let vr_u32 = vshrq_n_u32(vreinterpretq_u32_f32(vr), 16);
            vst1_u16(res.as_mut_ptr().add(i) as *mut u16, vmovn_u32(vr_u32));
        }
    }
    add_bf16_scalar(&a[n4..], &b[n4..], &mut res[n4..]);
}

fn add_bf16_scalar(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    for i in 0..a.len() {
        res[i] = half::bf16::from_f32(a[i].to_f32() + b[i].to_f32());
    }
}
