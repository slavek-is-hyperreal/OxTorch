#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Tier 2: SSE Interleaved 1x4.
#[cfg(target_arch = "x86_64")]
pub unsafe fn execute_bit_linear_sse(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    let mask = _mm_set1_epi8(0x03);
    let ones_i16 = _mm_set1_epi16(1);

    let a_sum: i32 = activations_i8.iter().map(|&x| x as i32).sum();

    let n_groups = m / 4;
    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();
        let mut acc2 = _mm_setzero_si128();
        let mut acc3 = _mm_setzero_si128();

        for kk in (0..k).step_by(16) {
            let act_vec = _mm_loadu_si128(activations_i8.as_ptr().add(kk) as *const __m128i);
            let w_packed = _mm_loadu_si128(weights_packed.as_ptr().add(rg * k + kk) as *const __m128i);

            let v0 = _mm_and_si128(_mm_srli_epi16(w_packed, 6), mask);
            let v1 = _mm_and_si128(_mm_srli_epi16(w_packed, 4), mask);
            let v2 = _mm_and_si128(_mm_srli_epi16(w_packed, 2), mask);
            let v3 = _mm_and_si128(w_packed, mask);

            acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_maddubs_epi16(v0, act_vec), ones_i16));
            acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_maddubs_epi16(v1, act_vec), ones_i16));
            acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_maddubs_epi16(v2, act_vec), ones_i16));
            acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_maddubs_epi16(v3, act_vec), ones_i16));
        }

        fn hsum(v: __m128i) -> i32 {
            let mut arr = [0i32; 4];
            unsafe { _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, v); }
            arr[0] + arr[1] + arr[2] + arr[3]
        }

        let sums = [hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3)];
        for i in 0..4 {
            output[r_out_base + i] = (sums[i] - a_sum) as f32 * scales[r_out_base + i];
        }
    }
}
