//! 512-bit AVX-512 Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Utilizes 32 ZMM registers, per-lane predication (k-registers), and Non-Temporal Stores.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    let abs_mask = _mm512_set1_ps(f32::from_bits(0x7FFFFFFF));
    let sign_bit = _mm512_set1_ps(f32::from_bits(0x80000000));
    
    // Polynomial Constants (Minimax Remez)
    let c0 = _mm512_set1_ps(0.99978784);
    let c1 = _mm512_set1_ps(-0.32580840);
    let c2 = _mm512_set1_ps(0.15557865);
    let c3 = _mm512_set1_ps(-0.04432655);
    
    let pi = _mm512_set1_ps(core::f32::consts::PI);
    let pi_2 = _mm512_set1_ps(core::f32::consts::FRAC_PI_2);

    // AVX-512 Unrolling Strategy: 4 vectors (64 floats) per iteration
    // Utilizing the 32 ZMM registers to keep all constants and intermediate state hot.
    while i + 63 < n {
        // Load Block
        let (y0, x0) = (_mm512_loadu_ps(y.as_ptr().add(i)), _mm512_loadu_ps(x.as_ptr().add(i)));
        let (y1, x1) = (_mm512_loadu_ps(y.as_ptr().add(i + 16)), _mm512_loadu_ps(x.as_ptr().add(i + 16)));
        let (y2, x2) = (_mm512_loadu_ps(y.as_ptr().add(i + 32)), _mm512_loadu_ps(x.as_ptr().add(i + 32)));
        let (y3, x3) = (_mm512_loadu_ps(y.as_ptr().add(i + 48)), _mm512_loadu_ps(x.as_ptr().add(i + 48)));

        // --- Domain Reduction ---
        let (ya0, xa0) = (_mm512_and_ps(y0, abs_mask), _mm512_and_ps(x0, abs_mask));
        let (ya1, xa1) = (_mm512_and_ps(y1, abs_mask), _mm512_and_ps(x1, abs_mask));
        let (ya2, xa2) = (_mm512_and_ps(y2, abs_mask), _mm512_and_ps(x2, abs_mask));
        let (ya3, xa3) = (_mm512_and_ps(y3, abs_mask), _mm512_and_ps(x3, abs_mask));

        // Predicate masks: True if ya > xa
        let k_swap0 = _mm512_cmp_ps_mask(ya0, xa0, _CMP_GT_OQ);
        let k_swap1 = _mm512_cmp_ps_mask(ya1, xa1, _CMP_GT_OQ);
        let k_swap2 = _mm512_cmp_ps_mask(ya2, xa2, _CMP_GT_OQ);
        let k_swap3 = _mm512_cmp_ps_mask(ya3, xa3, _CMP_GT_OQ);

        let a0 = _mm512_div_ps(_mm512_min_ps(ya0, xa0), _mm512_max_ps(ya0, xa0));
        let a1 = _mm512_div_ps(_mm512_min_ps(ya1, xa1), _mm512_max_ps(ya1, xa1));
        let a2 = _mm512_div_ps(_mm512_min_ps(ya2, xa2), _mm512_max_ps(ya2, xa2));
        let a3 = _mm512_div_ps(_mm512_min_ps(ya3, xa3), _mm512_max_ps(ya3, xa3));

        // --- Polynomial (Horner FMA) ---
        let sq0 = _mm512_mul_ps(a0, a0); let sq1 = _mm512_mul_ps(a1, a1);
        let sq2 = _mm512_mul_ps(a2, a2); let sq3 = _mm512_mul_ps(a3, a3);

        let mut p0 = _mm512_fmadd_ps(sq0, c3, c2);
        let mut p1 = _mm512_fmadd_ps(sq1, c3, c2);
        let mut p2 = _mm512_fmadd_ps(sq2, c3, c2);
        let mut p3 = _mm512_fmadd_ps(sq3, c3, c2);

        p0 = _mm512_fmadd_ps(sq0, p0, c1);
        p1 = _mm512_fmadd_ps(sq1, p1, c1);
        p2 = _mm512_fmadd_ps(sq2, p2, c1);
        p3 = _mm512_fmadd_ps(sq3, p3, c1);

        p0 = _mm512_fmadd_ps(sq0, p0, c0);
        p1 = _mm512_fmadd_ps(sq1, p1, c0);
        p2 = _mm512_fmadd_ps(sq2, p2, c0);
        p3 = _mm512_fmadd_ps(sq3, p3, c0);

        p0 = _mm512_mul_ps(a0, p0);
        p1 = _mm512_mul_ps(a1, p1);
        p2 = _mm512_mul_ps(a2, p2);
        p3 = _mm512_mul_ps(a3, p3);

        // --- Restore Quadrant using K-registers ---
        // 1. Swap if ya > xa
        p0 = _mm512_mask_blend_ps(k_swap0, p0, _mm512_sub_ps(pi_2, p0));
        p1 = _mm512_mask_blend_ps(k_swap1, p1, _mm512_sub_ps(pi_2, p1));
        p2 = _mm512_mask_blend_ps(k_swap2, p2, _mm512_sub_ps(pi_2, p2));
        p3 = _mm512_mask_blend_ps(k_swap3, p3, _mm512_sub_ps(pi_2, p3));

        // 2. Mirror if x < 0
        let k_xneg0 = _mm512_cmp_ps_mask(x0, _mm512_setzero_ps(), _CMP_LT_OQ);
        let k_xneg1 = _mm512_cmp_ps_mask(x1, _mm512_setzero_ps(), _CMP_LT_OQ);
        let k_xneg2 = _mm512_cmp_ps_mask(x2, _mm512_setzero_ps(), _CMP_LT_OQ);
        let k_xneg3 = _mm512_cmp_ps_mask(x3, _mm512_setzero_ps(), _CMP_LT_OQ);
        p0 = _mm512_mask_blend_ps(k_xneg0, p0, _mm512_sub_ps(pi, p0));
        p1 = _mm512_mask_blend_ps(k_xneg1, p1, _mm512_sub_ps(pi, p1));
        p2 = _mm512_mask_blend_ps(k_xneg2, p2, _mm512_sub_ps(pi, p2));
        p3 = _mm512_mask_blend_ps(k_xneg3, p3, _mm512_sub_ps(pi, p3));

        // 3. Negate based on original Y sign (bitwise XOR)
        p0 = _mm512_xor_ps(p0, _mm512_and_ps(y0, sign_bit));
        p1 = _mm512_xor_ps(p1, _mm512_and_ps(y1, sign_bit));
        p2 = _mm512_xor_ps(p2, _mm512_and_ps(y2, sign_bit));
        p3 = _mm512_xor_ps(p3, _mm512_and_ps(y3, sign_bit));

        // Store Non-Temporal (VMOVNTPS)
        _mm512_stream_ps(res.as_ptr().add(i) as *mut f32, p0);
        _mm512_stream_ps(res.as_ptr().add(i + 16) as *mut f32, p1);
        _mm512_stream_ps(res.as_ptr().add(i + 32) as *mut f32, p2);
        _mm512_stream_ps(res.as_ptr().add(i + 48) as *mut f32, p3);

        i += 64;
    }

    // Scalar Rest
    for r in i..n {
        let (yi, xi) = (y[r], x[r]);
        let ya = yi.abs(); let xa = xi.abs();
        let (a, swp) = if ya > xa { (xa/ya, true) } else { (ya/xa, false) };
        let s = a * a;
        let mut p = a * (0.99978784 + s * (-0.32580840 + s * (0.15557865 + s * -0.04432655)));
        if swp { p = core::f32::consts::FRAC_PI_2 - p; }
        if xi < 0.0 { p = core::f32::consts::PI - p; }
        if yi < 0.0 { p = -p; }
        res[r] = p;
    }
    
    _mm_sfence();
}
