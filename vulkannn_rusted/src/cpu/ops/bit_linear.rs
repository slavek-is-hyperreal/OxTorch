use rayon::prelude::*;
use crate::tensor::DataType;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// High-performance BitNet Linear layer dispatcher.
/// Supports 4 tiers: Scalar (SWAR), AVX1 (SSSE3), AVX2, and AVX512.
pub fn bit_linear_f32(m: usize, k: usize, n: usize, a: &[i8], b: &[u8], s: &[f32], c: &mut [f32], dtype: DataType) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            // Future Tier 4
            bit_linear_scalar(m, k, n, a, b, s, c, dtype); 
        } else if is_x86_feature_detected!("avx2") {
            bit_linear_avx2(m, k, n, a, b, s, c, dtype);
        } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("ssse3") {
            bit_linear_avx1(m, k, n, a, b, s, c, dtype);
        } else {
            bit_linear_scalar(m, k, n, a, b, s, c, dtype);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        bit_linear_scalar(m, k, n, a, b, s, c, dtype);
    }
}

/// Tier 1: Scalar fallback / SWAR-lite.
/// Processes BitNet2 (4 trits/byte) or BitNet1.6 (5 trits/byte).
pub fn bit_linear_scalar(_m: usize, k: usize, n: usize, a: &[i8], b: &[u8], s: &[f32], c: &mut [f32], dtype: DataType) {
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k .. (i + 1) * k];
        for j in 0..n {
            let mut sum: i32 = 0;
            if dtype == DataType::BitNet2 {
                let b_row = &b[j * ((k + 3) / 4) .. (j + 1) * ((k + 3) / 4)];
                for kk in 0..k {
                    let byte = b_row[kk / 4];
                    let shift = (kk % 4) * 2;
                    let val = (byte >> shift) & 0x03;
                    let w = (val as i8) - 1; // 0->-1, 1->0, 2->1
                    sum += (a_row[kk] as i32) * (w as i32);
                }
            } else {
                let b_row = &b[j * ((k + 4) / 5) .. (j + 1) * ((k + 4) / 5)];
                for kk in 0..k {
                    let byte = b_row[kk / 5];
                    let mut b_val = byte;
                    for _ in 0..(kk % 5) { b_val /= 3; }
                    let w = ((b_val % 3) as i8) - 1;
                    sum += (a_row[kk] as i32) * (w as i32);
                }
            }
            row[j] = (sum as f32) * s[j];
        }
    });
}

/// Tier 2: AVX1 / SSSE3 (Optimized for Ivy Bridge i5-3450).
/// Uses the "Shifted-Sum" trick: sum(W*X) = sum((W+1)*X) - sum(X).
#[cfg(target_arch = "x86_64")]
pub fn bit_linear_avx1(m: usize, k: usize, n: usize, a: &[i8], b: &[u8], s: &[f32], c: &mut [f32], dtype: DataType) {
    if dtype != DataType::BitNet2 {
        return bit_linear_scalar(m, k, n, a, b, s, c, dtype);
    }

    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k .. (i + 1) * k];
        
        // Pre-calculate sum(X) for the shifted-sum trick.
        let mut row_sum_x = 0i32;
        for &x in a_row { row_sum_x += x as i32; }

        for j in 0..n {
            let b_row = &b[j * ((k + 3) / 4) .. (j + 1) * ((k + 3) / 4)];
            let mut acc: i32;
            
            unsafe {
                let mut sum_v_x = _mm_setzero_si128();
                let v_one = _mm_set1_epi16(1);
                
                let mut kk = 0;
                // Process 16 elements at a time (4 bytes of packed weights, 16 bytes of activations)
                while kk + 16 <= k {
                    // 1. Load 16 activations (128-bit)
                    let x_vec = _mm_loadu_si128(a_row.as_ptr().add(kk) as *const __m128i);
                    
                    // 2. Load 4 bytes of weights and unpack into 16 bytes of {0, 1, 2}
                    let b_packed = *(b_row.as_ptr().add(kk / 4) as *const u32);
                    
                    // Manual Unpacking (vpshufb could work with a LUT, but shifts are portable)
                    let mut b_unpacked = [0u8; 16];
                    for idx in 0..16 {
                        b_unpacked[idx] = ((b_packed >> ((idx % 4) * 2 + (idx/4)*8)) & 0x03) as u8;
                    }
                    let v_vec = _mm_loadu_si128(b_unpacked.as_ptr() as *const __m128i);

                    // 3. maddubs: (u8 * i8) -> i16 (pairs summed)
                    let m_i16 = _mm_maddubs_epi16(v_vec, x_vec);
                    
                    // 4. Horizontal sum i16 -> i32
                    sum_v_x = _mm_add_epi32(sum_v_x, _mm_madd_epi16(m_i16, v_one));
                    
                    kk += 16;
                }
                
                // Vertical sum of the 4 partial sums in sum_v_x
                let mut tmp = [0i32; 4];
                _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum_v_x);
                acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                // Remainder
                for rem_kk in kk..k {
                    let w_unpacked = ((b_row[rem_kk / 4] >> ((rem_kk % 4) * 2)) & 0x03) as i32;
                    acc += (a_row[rem_kk] as i32) * w_unpacked;
                }
                
                // Shifted-Sum Correction: sum(W*X) = sum((W+1)*X) - sum(X)
                acc -= row_sum_x;
            }
            row[j] = (acc as f32) * s[j];
        }
    });
}

/// Tier 3: AVX2.
#[cfg(target_arch = "x86_64")]
pub fn bit_linear_avx2(m: usize, k: usize, n: usize, a: &[i8], b: &[u8], s: &[f32], c: &mut [f32], dtype: DataType) {
    // Placeholder: Similar to AVX1 but with 256-bit registers and _mm256_maddubs_epi16.
    bit_linear_avx1(m, k, n, a, b, s, c, dtype);
}
