#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Serial AVX implementation for BF16 subtraction.
/// Optimized as a "Leaf Kernel" for MSTS/Rayon.
#[target_feature(enable = "avx")]
pub unsafe fn sub_bf16_avx_serial(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    let n8 = (n / 8) * 8;
    let zero = _mm_setzero_si128();

    for i in (0..n8).step_by(8) {
        let ba = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let bb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        
        // Upcast to F32
        let va = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero, ba)),
            _mm_unpackhi_epi16(zero, ba)
        ));
        let vb = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero, bb)),
            _mm_unpackhi_epi16(zero, bb)
        ));
        
        let vr = _mm256_sub_ps(va, vb);
        
        // Downcast to BF16
        let res_si = _mm256_castps_si256(vr);
        let h_lo = _mm_srli_epi32(_mm256_castsi256_si128(res_si), 16);
        let h_hi = _mm_srli_epi32(_mm256_extractf128_si256::<1>(res_si), 16);
        let packed = _mm_packus_epi32(h_lo, h_hi);
        
        _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, packed);
    }

    if n > n8 {
        for i in n8..n {
            res[i] = half::bf16::from_f32(a[i].to_f32() - b[i].to_f32());
        }
    }
}
