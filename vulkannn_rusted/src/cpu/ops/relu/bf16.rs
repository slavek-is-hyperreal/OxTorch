use half;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

pub fn relu_bf16(src: &[half::bf16], dst: &mut [half::bf16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        return unsafe { relu_bf16_avx(src, dst) };
    }
    for (o, &i) in dst.iter_mut().zip(src.iter()) {
        *o = if i.to_f32() > 0.0 { i } else { half::bf16::ZERO };
    }
}

pub fn relu_bf16_inplace(buf: &mut [half::bf16]) {
    let len = buf.len();
    let ptr = buf.as_mut_ptr();
    let src = unsafe { std::slice::from_raw_parts(ptr, len) };
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    relu_bf16(src, dst);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
unsafe fn relu_bf16_avx(src: &[half::bf16], dst: &mut [half::bf16]) {
    let n8 = (src.len() / 8) * 8;
    let zero = _mm256_setzero_ps();
    let zero128 = _mm_setzero_si128();
    for i in (0..n8).step_by(8) {
        let b_raw = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, b_raw)),
            _mm_unpackhi_epi16(zero128, b_raw)
        ));
        let res_f = _mm256_max_ps(f_vec, zero);
        let res_si = _mm256_castps_si256(res_f);
        let h_lo = _mm_srli_epi32(_mm256_castsi256_si128(res_si), 16);
        let h_hi = _mm_srli_epi32(_mm256_extractf128_si256(res_si, 1), 16);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, _mm_packus_epi32(h_lo, h_hi));
    }
    for j in n8..src.len() {
        dst[j] = if src[j].to_f32() > 0.0 { src[j] } else { half::bf16::ZERO };
    }
}
