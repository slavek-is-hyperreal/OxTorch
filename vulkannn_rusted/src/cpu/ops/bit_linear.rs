/// High-performance BitNet Linear layer dispatcher.
pub fn bit_linear_f32(m: usize, k: usize, n: usize, a: &[i8], b: &[u8], s: &[f32], c: &mut [f32], _dtype: crate::tensor::DataType) {
    #[cfg(target_arch = "x86_64")]
    {
        use rayon::prelude::*;
        
        // Loop over sequence length M (Prefill support)
        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let a_row = &a[i * k .. (i + 1) * k];
            unsafe {
                if is_x86_feature_detected!("avx2") {
                    crate::cpu::ops::bitnet_lut::execute_bit_linear_avx2(n, k, b, a_row, s, row);
                } else if is_x86_feature_detected!("ssse3") {
                    crate::cpu::ops::bitnet_lut::execute_bit_linear_sse(n, k, b, a_row, s, row);
                } else {
                    crate::cpu::ops::bitnet_lut::execute_bit_linear_scalar(n, k, b, a_row, s, row);
                }
            }
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..m {
             let a_row = &a[i * k .. (i + 1) * k];
             let row = &mut c[i * n .. (i + 1) * n];
             crate::cpu::ops::bitnet_lut::execute_bit_linear_scalar(n, k, b, a_row, s, row);
        }
    }
}
