#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
use std::arch::x86_64::*;

/// Tier 4: AVX512 Interleaved 1x4.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
pub unsafe fn execute_bit_linear_avx512(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    let mask = _mm512_set1_epi8(0x03);
    let ones_i16 = _mm512_set1_epi16(1);

    let a_sum: i32 = activations_i8.iter().map(|&x| x as i32).sum();

    let n_groups = m / 4;
    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut acc2 = _mm512_setzero_si512();
        let mut acc3 = _mm512_setzero_si512();

        for kk in (0..k).step_by(64) {
            let act_vec = _mm512_loadu_si512(activations_i8.as_ptr().add(kk) as *const i32);
            let w_packed = _mm512_loadu_si512(weights_packed.as_ptr().add(rg * k + kk) as *const i32);

            let v0 = _mm512_and_si512(_mm512_srli_epi16(w_packed, 6), mask);
            let v1 = _mm512_and_si512(_mm512_srli_epi16(w_packed, 4), mask);
            let v2 = _mm512_and_si512(_mm512_srli_epi16(w_packed, 2), mask);
            let v3 = _mm512_and_si512(w_packed, mask);

            // maddubs handles i8 * u8 -> i16
            // madd handles i16 * i16 -> i32 (summing pairs)
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(_mm512_maddubs_epi16(v0, act_vec), ones_i16));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(_mm512_maddubs_epi16(v1, act_vec), ones_i16));
            acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(_mm512_maddubs_epi16(v2, act_vec), ones_i16));
            acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(_mm512_maddubs_epi16(v3, act_vec), ones_i16));
        }

        fn hsum512(v: __m512i) -> i32 {
            let mut arr = [0i32; 16];
            unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut i32, v); }
            arr.iter().sum()
        }

        let sums = [hsum512(acc0), hsum512(acc1), hsum512(acc2), hsum512(acc3)];
        for i in 0..4 {
            output[r_out_base + i] = (sums[i] - a_sum) as f32 * scales[r_out_base + i];
        }
    }
}

/// Fallback for non-AVX512 builds
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
pub unsafe fn execute_bit_linear_avx512(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    crate::cpu::ops::bitnet::avx2::execute_bit_linear_avx2(m, k, weights_packed, activations_i8, scales, output);
    #[cfg(not(target_arch = "x86_64"))]
    crate::cpu::ops::bitnet::scalar::execute_bit_linear_scalar(m, k, weights_packed, activations_i8, scales, output);
}
