use half;
#[allow(unused_imports)]
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

pub fn matmul_bf16(m: usize, k: usize, n: usize, a: &[half::bf16], b: &[half::bf16], c: &mut [half::bf16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        unsafe { matmul_bf16_avx(m, k, n, a, b, c); return; }
    }
    
    // Scalar fallback
    c.fill(half::bf16::ZERO);
    for i in 0..m {
        for kk in 0..k {
            let aval = a[i * k + kk].to_f32();
            for j in 0..n {
                let mut res = c[i * n + j].to_f32();
                res += aval * b[kk * n + j].to_f32();
                c[i * n + j] = half::bf16::from_f32(res);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
unsafe fn matmul_bf16_avx(_m: usize, k: usize, n: usize, a: &[half::bf16], b: &[half::bf16], c: &mut [half::bf16]) {
    c.fill(half::bf16::ZERO);
    let zero128 = _mm_setzero_si128();
    c.par_chunks_mut(n).enumerate().for_each(|(i, row_c)| {
        for kk in 0..k {
            let a_f32 = a[i * k + kk].to_f32();
            let a_vec = _mm256_set1_ps(a_f32);
            let mut j = 0;
            while j + 8 <= n {
                let ba = _mm_loadu_si128(b.as_ptr().add(kk * n + j) as *const __m128i);
                let b_v = _mm256_castps_si256(_mm256_insertf128_si256::<1>(
                    _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, ba)),
                    _mm_unpackhi_epi16(zero128, ba)
                ));
                let b_ps = _mm256_castsi256_ps(b_v);
                
                let ca = _mm_loadu_si128(row_c.as_ptr().add(j) as *const __m128i);
                let c_v = _mm256_castps_si256(_mm256_insertf128_si256::<1>(
                    _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, ca)),
                    _mm_unpackhi_epi16(zero128, ca)
                ));
                let c_ps = _mm256_castsi256_ps(c_v);
                
                let res_f = _mm256_add_ps(c_ps, _mm256_mul_ps(a_vec, b_ps));
                let res_si = _mm256_castps_si256(res_f);
                let h_lo = _mm_srli_epi32(_mm256_castsi256_si128(res_si), 16);
                let h_hi = _mm_srli_epi32(_mm256_extractf128_si256(res_si, 1), 16);
                _mm_storeu_si128(row_c.as_mut_ptr().add(j) as *mut __m128i, _mm_packus_epi32(h_lo, h_hi));
                
                j += 8;
            }
            while j < n {
                row_c[j] = half::bf16::from_f32(row_c[j].to_f32() + a_f32 * b[kk * n + j].to_f32());
                j += 1;
            }
        }
    });
}
