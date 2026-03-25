#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Tier 1: Scalar fallback / SWAR-lite (Interleaved 1x4).
pub fn execute_bit_linear_scalar(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    // weights_packed is [m/4, k] - row-interleaved 1x4
    let n_groups = m / 4;
    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut sums = [0i32; 4];
        
        for kk in 0..k {
            let act = activations_i8[kk] as i32;
            let packed = weights_packed[rg * k + kk];
            
            // Unpack 4 rows: LSB-first (matches safetensors format)
            // bits[1:0]=row*4+0, bits[3:2]=row*4+1, bits[5:4]=row*4+2, bits[7:6]=row*4+3
            sums[0] += act * (((packed >> 0) & 0x03) as i32 - 1);
            sums[1] += act * (((packed >> 2) & 0x03) as i32 - 1);
            sums[2] += act * (((packed >> 4) & 0x03) as i32 - 1);
            sums[3] += act * (((packed >> 6) & 0x03) as i32 - 1);
        }
        
        for i in 0..4 {
            output[r_out_base + i] = (sums[i] as f32) * scales[r_out_base + i];
        }
    }
}

/// Tier 2: SSE (AVX1) Interleaved 1x4.
#[cfg(target_arch = "x86_64")]
pub unsafe fn execute_bit_linear_sse(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    if !is_x86_feature_detected!("ssse3") {
        return execute_bit_linear_scalar(m, k, weights_packed, activations_i8, scales, output);
    }

    let mask = _mm_set1_epi8(0x03);
    let ones_u8 = _mm_set1_epi8(1);
    let ones_i16 = _mm_set1_epi16(1);
    let bias_128 = _mm_set1_epi8(-128);

    let n_groups = m / 4;
    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();
        let mut acc2 = _mm_setzero_si128();
        let mut acc3 = _mm_setzero_si128();
        
        let mut wsum0 = _mm_setzero_si128();
        let mut wsum1 = _mm_setzero_si128();
        let mut wsum2 = _mm_setzero_si128();
        let mut wsum3 = _mm_setzero_si128();

        for kk in (0..k).step_by(16) {
            let act_vec = _mm_loadu_si128(activations_i8.as_ptr().add(kk) as *const __m128i);
            let act_u8 = _mm_add_epi8(act_vec, bias_128); // X + 128
            
            let w_packed = _mm_loadu_si128(weights_packed.as_ptr().add(rg * k + kk) as *const __m128i);
            
            // Slicing LSB-first: bits[1:0]=row0, bits[3:2]=row1, bits[5:4]=row2, bits[7:6]=row3
            let v0 = _mm_and_si128(w_packed, mask);
            let v1 = _mm_and_si128(_mm_srli_epi16(w_packed, 2), mask);
            let v2 = _mm_and_si128(_mm_srli_epi16(w_packed, 4), mask);
            let v3 = _mm_and_si128(_mm_srli_epi16(w_packed, 6), mask);
            
            let ws0 = _mm_sub_epi8(v0, ones_u8); // W in {-1, 0, 1}
            let ws1 = _mm_sub_epi8(v1, ones_u8);
            let ws2 = _mm_sub_epi8(v2, ones_u8);
            let ws3 = _mm_sub_epi8(v3, ones_u8);

            acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_maddubs_epi16(act_u8, ws0), ones_i16));
            acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_maddubs_epi16(act_u8, ws1), ones_i16));
            acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_maddubs_epi16(act_u8, ws2), ones_i16));
            acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_maddubs_epi16(act_u8, ws3), ones_i16));

            wsum0 = _mm_add_epi32(wsum0, _mm_madd_epi16(_mm_maddubs_epi16(ones_u8, ws0), ones_i16));
            wsum1 = _mm_add_epi32(wsum1, _mm_madd_epi16(_mm_maddubs_epi16(ones_u8, ws1), ones_i16));
            wsum2 = _mm_add_epi32(wsum2, _mm_madd_epi16(_mm_maddubs_epi16(ones_u8, ws2), ones_i16));
            wsum3 = _mm_add_epi32(wsum3, _mm_madd_epi16(_mm_maddubs_epi16(ones_u8, ws3), ones_i16));
        }

        fn hsum(v: __m128i) -> i32 {
            let mut arr = [0i32; 4];
            unsafe { _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, v); }
            arr[0] + arr[1] + arr[2] + arr[3]
        }

        let sums = [hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3)];
        let wsums = [hsum(wsum0), hsum(wsum1), hsum(wsum2), hsum(wsum3)];

        for i in 0..4 {
            let final_sum = sums[i] - 128 * wsums[i];
            output[r_out_base + i] = (final_sum as f32) * scales[r_out_base + i];
        }
    }
}

/// Tier 3: AVX2 Interleaved 1x4.
#[cfg(target_arch = "x86_64")]
pub unsafe fn execute_bit_linear_avx2(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    if !is_x86_feature_detected!("avx2") {
        return execute_bit_linear_sse(m, k, weights_packed, activations_i8, scales, output);
    }
    
    let mask = _mm256_set1_epi8(0x03);
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);
    let bias_128 = _mm256_set1_epi8(-128);

    let n_groups = m / 4;
    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        
        let mut wsum0 = _mm256_setzero_si256();
        let mut wsum1 = _mm256_setzero_si256();
        let mut wsum2 = _mm256_setzero_si256();
        let mut wsum3 = _mm256_setzero_si256();

        for kk in (0..k).step_by(32) {
            let act_vec = _mm256_loadu_si256(activations_i8.as_ptr().add(kk) as *const __m256i);
            let act_u8 = _mm256_add_epi8(act_vec, bias_128);
            
            let w_packed = _mm256_loadu_si256(weights_packed.as_ptr().add(rg * k + kk) as *const __m256i);
            
            // LSB-first: bits[1:0]=row0, bits[3:2]=row1, bits[5:4]=row2, bits[7:6]=row3
            let v0 = _mm256_and_si256(w_packed, mask);
            let v1 = _mm256_and_si256(_mm256_srli_epi16(w_packed, 2), mask);
            let v2 = _mm256_and_si256(_mm256_srli_epi16(w_packed, 4), mask);
            let v3 = _mm256_and_si256(_mm256_srli_epi16(w_packed, 6), mask);
            
            let ws0 = _mm256_sub_epi8(v0, ones_u8);
            let ws1 = _mm256_sub_epi8(v1, ones_u8);
            let ws2 = _mm256_sub_epi8(v2, ones_u8);
            let ws3 = _mm256_sub_epi8(v3, ones_u8);

            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(act_u8, ws0), ones_i16));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(act_u8, ws1), ones_i16));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(act_u8, ws2), ones_i16));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(act_u8, ws3), ones_i16));

            wsum0 = _mm256_add_epi32(wsum0, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, ws0), ones_i16));
            wsum1 = _mm256_add_epi32(wsum1, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, ws1), ones_i16));
            wsum2 = _mm256_add_epi32(wsum2, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, ws2), ones_i16));
            wsum3 = _mm256_add_epi32(wsum3, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, ws3), ones_i16));
        }

        fn hsum256(v: __m256i) -> i32 {
            let mut arr = [0i32; 8];
            unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, v); }
            arr.iter().sum()
        }

        let sums = [hsum256(acc0), hsum256(acc1), hsum256(acc2), hsum256(acc3)];
        let wsums = [hsum256(wsum0), hsum256(wsum1), hsum256(wsum2), hsum256(wsum3)];

        for i in 0..4 {
            let final_sum = sums[i] - 128 * wsums[i];
            output[r_out_base + i] = (final_sum as f32) * scales[r_out_base + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_linear_parity() {
        let m = 4;
        let k = 32;
        let mut weights = vec![0i8; m * k];
        for i in 0..m {
            for j in 0..k {
                weights[i * k + j] = ((i + j) % 3) as i8 - 1;
            }
        }
        
        let activations = vec![127i8; k];
        let scales = vec![1.0; m];
        let mut output = vec![0.0; m];

        // Reference
        let mut ref_out = vec![0.0; m];
        for i in 0..m {
            let mut s = 0i32;
            for j in 0..k {
                s += (weights[i * k + j] as i32) * (activations[j] as i32);
            }
            ref_out[i] = s as f32;
        }

        // Pack LSB-first (matches safetensors format)
        let mut packed = vec![0u8; (m * k) / 4];
        for kk in 0..k {
            let q0 = (weights[0 * k + kk] + 1) as u8;
            let q1 = (weights[1 * k + kk] + 1) as u8;
            let q2 = (weights[2 * k + kk] + 1) as u8;
            let q3 = (weights[3 * k + kk] + 1) as u8;
            packed[kk] = (q0 << 0) | (q1 << 2) | (q2 << 4) | (q3 << 6);
        }

        // Scalar
        execute_bit_linear_scalar(m, k, &packed, &activations, &scales, &mut output);
        for i in 0..m {
            assert!((output[i] - ref_out[i]).abs() < 1e-4);
        }

        // SSE
        #[cfg(target_arch = "x86_64")]
        unsafe {
            execute_bit_linear_sse(m, k, &packed, &activations, &scales, &mut output);
            for i in 0..m {
                assert!((output[i] - ref_out[i]).abs() < 1e-4);
            }
        }

        // AVX2
        #[cfg(target_arch = "x86_64")]
        unsafe {
            execute_bit_linear_avx2(m, k, &packed, &activations, &scales, &mut output);
            for i in 0..m {
                assert!((output[i] - ref_out[i]).abs() < 1e-4);
            }
        }
    }
}
