#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise multiplication for I8 tensors with multi-architecture support (Saturating).
pub fn mul_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    const PAR_THRESHOLD: usize = 4_000_000;
    if a.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            mul_i8_serial(ac, bc, rc);
        });
    } else {
        mul_i8_serial(a, b, res);
    }
}

fn mul_i8_serial(a: &[i8], b: &[i8], res: &mut [i8]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            return unsafe { mul_i8_avx2(a, b, res) };
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return mul_i8_neon(a, b, res);
    }

    add_i8_scalar(a, b, res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_i8_avx2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n32 = (a.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        
        let lo_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let hi_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let lo_b = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let hi_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        
        let prod_lo = _mm256_mullo_epi16(lo_a, lo_b);
        let prod_hi = _mm256_mullo_epi16(hi_a, hi_b);
        
        let res_vec = _mm256_packs_epi16(prod_lo, prod_hi);
        let res_ordered = _mm256_permute4x64_epi64(res_vec, 0xD8); 
        _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, res_ordered);
    }
    add_i8_scalar(&a[n32..], &b[n32..], &mut res[n32..]);
}

#[cfg(target_arch = "aarch64")]
fn mul_i8_neon(a: &[i8], b: &[i8], res: &mut [i8]) {
    use std::arch::aarch64::*;
    let n16 = (a.len() / 16) * 16;
    for i in (0..n16).step_by(16) {
        unsafe {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            
            // Neon doesn't have 8x8->8 saturating mul directly, need to widen to 16
            let lo_a = vmovl_s8(vget_low_s8(va));
            let hi_a = vmovl_s8(vget_high_s8(va));
            let lo_b = vmovl_s8(vget_low_s8(vb));
            let hi_b = vmovl_s8(vget_high_s8(vb));
            
            let prod_lo = vmulq_s16(lo_a, lo_b);
            let prod_hi = vmulq_s16(hi_a, hi_b);
            
            vst1q_s8(res.as_mut_ptr().add(i), vcombine_s8(vqmovn_s16(prod_lo), vqmovn_s16(prod_hi)));
        }
    }
    add_i8_scalar(&a[n16..], &b[n16..], &mut res[n16..]);
}

fn add_i8_scalar(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = a[i].saturating_mul(b[i]);
    }
}
