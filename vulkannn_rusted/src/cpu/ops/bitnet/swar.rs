/// Tier 1.5: SWAR (SIMD Within A Register) for Scalar CPUs.
/// This implementation allows for processing multiple 2-bit weights in parallel using
/// standard 64-bit integer registers.
pub fn execute_bit_linear_swar(m: usize, k: usize, weights_packed: &[u8], activations_i8: &[i8], scales: &[f32], output: &mut [f32]) {
    // We treat every group of 4 rows.
    let n_groups = m / 4;
    let a_sum: i32 = activations_i8.iter().map(|&x| x as i32).sum();

    for rg in 0..n_groups {
        let r_out_base = rg * 4;
        let mut dots = [0i32; 4];

        // Process 8 weights at a time in each of the 4 rows (8 bytes * 8 bits = 64 bits = 8 activations)
        // Except weights_packed is [M/4, K]. Each byte in K is 4 rows.
        // So we process 8 columns (8 bytes) at once.
        let mut kk = 0;
        while kk + 8 <= k {
            // Load 8 columns of weights for these 4 rows
            let w_u64 = u64::from_le_bytes(weights_packed[rg * k + kk .. rg * k + kk + 8].try_into().unwrap());
            
            // Extract bits for each row across the 8 columns
            let r0_bits = (w_u64 >> 6) & 0x0303030303030303;
            let r1_bits = (w_u64 >> 4) & 0x0303030303030303;
            let r2_bits = (w_u64 >> 2) & 0x0303030303030303;
            let r3_bits = w_u64 & 0x0303030303030303;

            // Sequential for now, but using 64-bit reads reduces memory operations.
            // Further optimization would involve splitting r_bits into individual bytes and processing.
            // For a true SWAR, we'd need to pack activations into u64 as well.
            for i in 0..8 {
                let act = activations_i8[kk + i] as i32;
                dots[0] += ((r0_bits >> (i * 8)) & 0xFF) as i32 * act;
                dots[1] += ((r1_bits >> (i * 8)) & 0xFF) as i32 * act;
                dots[2] += ((r2_bits >> (i * 8)) & 0xFF) as i32 * act;
                dots[3] += ((r3_bits >> (i * 8)) & 0xFF) as i32 * act;
            }
            kk += 8;
        }

        // Remainder columns
        while kk < k {
            let byte = weights_packed[rg * k + kk];
            let act = activations_i8[kk] as i32;
            dots[0] += ((byte >> 6) & 0x03) as i32 * act;
            dots[1] += ((byte >> 4) & 0x03) as i32 * act;
            dots[2] += ((byte >> 2) & 0x03) as i32 * act;
            dots[3] += (byte & 0x03) as i32 * act;
            kk += 1;
        }

        for i in 0..4 {
            output[r_out_base + i] = (dots[i] - a_sum) as f32 * scales[r_out_base + i];
        }
    }

    // Handled remainder m rows
    if m % 4 != 0 {
        let start_row = n_groups * 4;
        for r_idx in start_row..m {
            let mut dot = 0i32;
            let row_offset = r_idx % 4;
            for kk in 0..k {
                let byte = weights_packed[n_groups * k + kk];
                let q = (byte >> (6 - 2 * row_offset)) & 0x03;
                dot += (q as i32) * (activations_i8[kk] as i32);
            }
            output[r_idx] = (dot - a_sum) as f32 * scales[r_idx];
        }
    }
}
