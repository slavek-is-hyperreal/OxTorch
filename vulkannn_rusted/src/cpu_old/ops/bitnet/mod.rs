pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod sse;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
pub mod swar;

use std::io::Write;

/// High-performance BitNet Linear layer dispatcher.
pub fn bit_linear_f32(m: usize, k: usize, n: usize, a: &[i8], b: &[u8], s: &[f32], c: &mut [f32], _dtype: crate::tensor::DataType) {
    #[cfg(target_arch = "x86_64")]
    {
        // Loop over sequence length M (batch size)
        for (i, row) in c.chunks_mut(n).enumerate() {
            if m > 1 {
                print!(".");
                std::io::stdout().flush().ok();
            }

            assert!(a.len() >= m * k, "A length short!");
            assert!(b.len() >= (n / 4) * k, "B length short! Needs N/4 * K");
            assert!(s.len() >= n, "S length short!");
            
            let a_row = &a[i * k .. (i + 1) * k];
            unsafe {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                    avx512::execute_bit_linear_avx512(n, k, b, a_row, s, row);
                } else if is_x86_feature_detected!("avx2") {
                    avx2::execute_bit_linear_avx2(n, k, b, a_row, s, row);
                } else if is_x86_feature_detected!("ssse3") {
                    sse::execute_bit_linear_sse(n, k, b, a_row, s, row);
                } else {
                    swar::execute_bit_linear_swar(n, k, b, a_row, s, row);
                }
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..m {
             let a_row = &a[i * k .. (i + 1) * k];
             let row = &mut c[i * n .. (i + 1) * n];
             swar::execute_bit_linear_swar(n, k, b, a_row, s, row);
        }
    }

    if m > 10 {
        println!("\n[BitLinear] Processed batch of {} rows", m);
    }
}
