#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use super::sse::execute_bit_linear_sse;

/// Tier 3: AVX2 Interleaved 1x4.
#[cfg(target_arch = "x86_64")]
pub unsafe fn execute_bit_linear_avx2(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    if !is_x86_feature_detected!("avx2") {
        return execute_bit_linear_sse(m, k, weights_packed, activations_i8, scales, output);
    }
    
    let mask = _mm256_set1_epi8(0x03);
    let ones_i16 = _mm256_set1_epi16(1);

    let a_sum: i32 = activations_i8.iter().map(|&x| x as i32).sum();

    let n_groups = m / 4;
    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for kk in (0..k).step_by(32) {
            let act_vec = _mm256_loadu_si256(activations_i8.as_ptr().add(kk) as *const __m256i);
            let w_packed = _mm256_loadu_si256(weights_packed.as_ptr().add(rg * k + kk) as *const __m256i);

            let v0 = _mm256_and_si256(_mm256_srli_epi16(w_packed, 6), mask);
            let v1 = _mm256_and_si256(_mm256_srli_epi16(w_packed, 4), mask);
            let v2 = _mm256_and_si256(_mm256_srli_epi16(w_packed, 2), mask);
            let v3 = _mm256_and_si256(w_packed, mask);

            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(v0, act_vec), ones_i16));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(v1, act_vec), ones_i16));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(v2, act_vec), ones_i16));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(v3, act_vec), ones_i16));
        }

        fn hsum256(v: __m256i) -> i32 {
            let mut arr = [0i32; 8];
            unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, v); }
            arr.iter().sum()
        }

        let sums = [hsum256(acc0), hsum256(acc1), hsum256(acc2), hsum256(acc3)];
        for i in 0..4 {
            output[r_out_base + i] = (sums[i] - a_sum) as f32 * scales[r_out_base + i];
        }
    }
}
