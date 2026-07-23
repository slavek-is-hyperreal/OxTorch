// ============================ KROK C AUDIT (2026-07) ========================
// AUDIT RESULT — read before touching bitnet in Wave 5.
//
// 1. ENCODING: this is NEITHER catalog C4a (RTN sign-magnitude two-plane
//    popcount) NOR C4b (pos/neg two-plane popcount). It is a THIRD scheme:
//    2-bit OFFSET-BINARY (q = weight + 1, so 00->-1, 01->0, 10->+1), computed by
//    a direct integer MAC `Σ q·act` with a bias correction `- a_sum` (a_sum =
//    Σ act), giving `Σ (q-1)·act = Σ w·act`. No popcount, no bit-planes. The
//    catalog C4 formulas DO NOT apply here; do not "port" them.
//
// 2. ROW-ORDER BUG (CONFIRMED by the golden test below): the packer
//    (tensor/conversion.rs::execute_to_bitnet, BitNet2) writes row0 of each
//    4-row group into the LOW bits `(q0<<0)|(q1<<2)|(q2<<4)|(q3<<6)` ("LSB-first,
//    matches safetensors"), but ALL compute kernels (swar, avx2, sse, scalar)
//    read row0 from the HIGH bits `(byte>>6)&3`. => within every group of 4
//    output rows, rows are REVERSED (0<->3, 1<->2). Silent wrong result, no
//    crash — exactly the class Krok C was meant to catch.
//    Golden proof: W rows [+1..],[-1..],[0..],[-1,+1..] · act -> expected
//    [16,-16,0,8] but kernels produce [8,0,-16,16].
//    OPEN QUESTION for the user: which layout is canonical? If models load
//    pre-packed from safetensors in MSB-first order, the compute kernels are
//    right and execute_to_bitnet is the buggy path; if execute_to_bitnet's
//    "matches safetensors" comment is correct, all compute kernels are reversed.
//    UNRESOLVED -> Wave 5 (matmul/bitnet/quant) is BLOCKED until decided.
// ===========================================================================

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

// ===========================================================================
// KROK C AUDIT (Wave-5 blocker). Golden test: pack a known sign-sensitive
// ternary weight matrix EXACTLY as tensor/conversion.rs::execute_to_bitnet does
// (q = w+1, LSB-first: (q0<<0)|(q1<<2)|(q2<<4)|(q3<<6), row0 in bits[0:1]), then
// run the SWAR kernel and compare each output row to the true dot Σ w·act·scale.
// The kernels read row0 from bits[6:7] (MSB) — this asserts whether pack/compute
// agree or the rows are permuted within each group of 4.
// ===========================================================================
#[cfg(test)]
mod krok_c_audit {
    use super::*;

    // Pack a [4, k] ternary (-1/0/+1) matrix into the SWAR byte layout, exactly
    // as execute_to_bitnet (BitNet2) does: q0 in the low bits, q3 in the high.
    fn pack_group(w: &[[i8; 8]; 4], k: usize) -> Vec<u8> {
        let mut packed = vec![0u8; k];
        for col in 0..k {
            let q = |r: usize| ((w[r][col] + 1).clamp(0, 2)) as u8;
            packed[col] = (q(0) << 0) | (q(1) << 2) | (q(2) << 4) | (q(3) << 6);
        }
        packed
    }

    #[test]
    fn swar_row_order_vs_packing() {
        let k = 8usize;
        // Sign-sensitive rows (the class that exposes encoding bugs):
        // row0 = all +1, row1 = all -1, row2 = all 0, row3 = alternating -1/+1.
        let w: [[i8; 8]; 4] = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ];
        let act: [i8; 8] = [2, 3, -1, 5, -4, 6, 7, -2];
        let _a_sum: i32 = act.iter().map(|&x| x as i32).sum();
        let scales = [1.0f32; 4];

        // True per-row dot: Σ w[r][c]*act[c].
        let mut expected = [0f32; 4];
        for r in 0..4 {
            let dot: i32 = (0..k).map(|c| w[r][c] as i32 * act[c] as i32).sum();
            expected[r] = dot as f32; // scale = 1
        }

        let packed = pack_group(&w, k);
        let mut out = [0f32; 4];
        // m=4, k=8. weights_packed layout [M/4, K] = [1, 8] = our `packed`.
        let act_i8: Vec<i8> = act.to_vec();
        execute_bit_linear_swar(4, k, &packed, &act_i8, &scales, &mut out);

        eprintln!("expected (pack row-order) = {:?}", expected);
        eprintln!("swar output              = {:?}", out);
        // If pack and compute AGREE, out == expected. If the rows are reversed
        // within the group (bug), out == [expected[3], expected[2], expected[1], expected[0]].
        let reversed = [expected[3], expected[2], expected[1], expected[0]];
        // FINDING LOCKED: the kernels currently produce the REVERSED order
        // (pack row0=LSB, compute row0=MSB). This assertion pins that CURRENT
        // (buggy) reality so the finding can't silently change; flip it to
        // `out == expected` once the row-order is reconciled in Wave 5.
        assert_eq!(out, reversed, "expected the known reversed order (see KROK C audit note)");
        assert_ne!(out, expected, "if this fires, the row order was fixed — update the audit note");
    }
}
