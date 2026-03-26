/// Tier 1: Scalar Reference / Fallback.
pub fn execute_bit_linear_scalar(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    // Calculate activation row sum for the -1 offset correction: sum(w*a) = sum((q-1)*a) = sum(q*a) - sum(a)
    let a_sum: i32 = activations_i8.iter().map(|&x| x as i32).sum();

    let n_groups = m / 4;
    for rg in 0..n_groups {
        for row_offset in 0..4 {
            let mut dot = 0i32;
            let r_idx = rg * 4 + row_offset;
            for kk in 0..k {
                let byte = weights_packed[rg * k + kk];
                let q = match row_offset {
                    0 => (byte >> 6) & 0x03,
                    1 => (byte >> 4) & 0x03,
                    2 => (byte >> 2) & 0x03,
                    3 => byte & 0x03,
                    _ => unreachable!(),
                };
                dot += (q as i32) * (activations_i8[kk] as i32);
            }
            output[r_idx] = (dot - a_sum) as f32 * scales[r_idx];
        }
    }

    // Handled remainder rows
    for r_idx in (n_groups * 4)..m {
        let mut dot = 0i32;
        for kk in 0..k {
            let byte = weights_packed[n_groups * k + kk];
            let row_offset = r_idx % 4;
            let q = match row_offset {
                0 => (byte >> 6) & 0x03,
                1 => (byte >> 4) & 0x03,
                2 => (byte >> 2) & 0x03,
                3 => byte & 0x03,
                _ => unreachable!(),
            };
            dot += (q as i32) * (activations_i8[kk] as i32);
        }
        output[r_idx] = (dot - a_sum) as f32 * scales[r_idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitnet_dot_sum() {
        let k = 128;
        let w = vec![0x55u8; k]; // All weights = 1 (bits 01 for all 4 rows in the group)
        let a = vec![1i8; k];
        let mut y = vec![0.0f32; 1];
        let scales = vec![1.0f32; 1];
        
        execute_bit_linear_scalar(1, k, &w, &a, &scales, &mut y);
        assert_eq!(y[0], 0.0);
        
        let w0 = vec![0x00u8; k]; // All weights = -1 (bits 00)
        execute_bit_linear_scalar(1, k, &w0, &a, &scales, &mut y);
        assert_eq!(y[0], -128.0);

        let w2 = vec![0xAAu8; k]; // All weights = +1 (bits 10 corresp to q=2)
        execute_bit_linear_scalar(1, k, &w2, &a, &scales, &mut y);
        assert_eq!(y[0], 128.0);
    }
}
