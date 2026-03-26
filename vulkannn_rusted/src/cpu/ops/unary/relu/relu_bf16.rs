#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;



/// Elementwise ReLU for BF16 tensors.
pub fn relu_bf16(in_buf: &[half::bf16], out_buf: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { relu_bf16_avx(in_buf, out_buf) }; }
    }
    for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32().max(0.0)); }
}

/// In-place ReLU for BF16 tensors.
pub fn relu_bf16_inplace(buf: &mut [half::bf16]) {
    for x in buf.iter_mut() { *x = half::bf16::from_f32(x.to_f32().max(0.0)); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn relu_bf16_avx(in_buf: &[half::bf16], out_buf: &mut [half::bf16]) {
    let zero = _mm256_setzero_ps(); let n8 = (in_buf.len() / 8) * 8;
    let zero128 = _mm_setzero_si128();
    for i in (0..n8).step_by(8) {
        let b_raw = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, b_raw)),
            _mm_unpackhi_epi16(zero128, b_raw)
        ));
        let f_res = _mm256_max_ps(f_vec, zero);
        // Pack back (approximate for ReLU it's simple as shifting)
        let f_int = _mm256_castps_si256(f_res);
        let lo = _mm256_extractf128_si256::<0>(f_int);
        let hi = _mm256_extractf128_si256::<1>(f_int);
        let pack = _mm_packus_epi32(_mm_srli_epi32(lo, 16), _mm_srli_epi32(hi, 16));
        _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, pack);
    }
    for i in n8..in_buf.len() { out_buf[i] = half::bf16::from_f32(in_buf[i].to_f32().max(0.0)); }
}
