//! Specialized AVX1 Implementation for Ivy Bridge (Intel Core 3rd Gen).
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Optimized for Port 1 affinity and 3rd-order Minimax Polynomial accuracy.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let sign_bit = _mm256_set1_ps(f32::from_bits(0x80000000));
    
    // Polynomial Constants (Remez Minimax)
    let c0 = _mm256_set1_ps(0.99978784);
    let c1 = _mm256_set1_ps(-0.32580840);
    let c2 = _mm256_set1_ps(0.15557865);
    let c3 = _mm256_set1_ps(-0.04432655);
    
    let pi = _mm256_set1_ps(core::f32::consts::PI);
    let pi_2 = _mm256_set1_ps(core::f32::consts::FRAC_PI_2);

    // Ivy Bridge Unrolling Strategy (8x unroll to saturate the limited Port 1 throughput)
    while i + 63 < n {
        // Load Block Y (64 floats)
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        let y1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
        let y2 = _mm256_loadu_ps(y.as_ptr().add(i + 16));
        let y3 = _mm256_loadu_ps(y.as_ptr().add(i + 24));
        let y4 = _mm256_loadu_ps(y.as_ptr().add(i + 32));
        let y5 = _mm256_loadu_ps(y.as_ptr().add(i + 40));
        let y6 = _mm256_loadu_ps(y.as_ptr().add(i + 48));
        let y7 = _mm256_loadu_ps(y.as_ptr().add(i + 56));

        // Load Block X (64 floats)
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let x2 = _mm256_loadu_ps(x.as_ptr().add(i + 16));
        let x3 = _mm256_loadu_ps(x.as_ptr().add(i + 24));
        let x4 = _mm256_loadu_ps(x.as_ptr().add(i + 32));
        let x5 = _mm256_loadu_ps(x.as_ptr().add(i + 40));
        let x6 = _mm256_loadu_ps(x.as_ptr().add(i + 48));
        let x7 = _mm256_loadu_ps(x.as_ptr().add(i + 56));

        // --- Domain Reduction ---
        let ya0 = _mm256_and_ps(y0, abs_mask); let xa0 = _mm256_and_ps(x0, abs_mask);
        let ya1 = _mm256_and_ps(y1, abs_mask); let xa1 = _mm256_and_ps(x1, abs_mask);
        let ya2 = _mm256_and_ps(y2, abs_mask); let xa2 = _mm256_and_ps(x2, abs_mask);
        let ya3 = _mm256_and_ps(y3, abs_mask); let xa3 = _mm256_and_ps(x3, abs_mask);
        let ya4 = _mm256_and_ps(y4, abs_mask); let xa4 = _mm256_and_ps(x4, abs_mask);
        let ya5 = _mm256_and_ps(y5, abs_mask); let xa5 = _mm256_and_ps(x5, abs_mask);
        let ya6 = _mm256_and_ps(y6, abs_mask); let xa6 = _mm256_and_ps(x6, abs_mask);
        let ya7 = _mm256_and_ps(y7, abs_mask); let xa7 = _mm256_and_ps(x7, abs_mask);

        let s_mask0 = _mm256_cmp_ps(ya0, xa0, _CMP_GT_OQ);
        let s_mask1 = _mm256_cmp_ps(ya1, xa1, _CMP_GT_OQ);
        let s_mask2 = _mm256_cmp_ps(ya2, xa2, _CMP_GT_OQ);
        let s_mask3 = _mm256_cmp_ps(ya3, xa3, _CMP_GT_OQ);
        let s_mask4 = _mm256_cmp_ps(ya4, xa4, _CMP_GT_OQ);
        let s_mask5 = _mm256_cmp_ps(ya5, xa5, _CMP_GT_OQ);
        let s_mask6 = _mm256_cmp_ps(ya6, xa6, _CMP_GT_OQ);
        let s_mask7 = _mm256_cmp_ps(ya7, xa7, _CMP_GT_OQ);

        let a0 = _mm256_div_ps(_mm256_min_ps(ya0, xa0), _mm256_max_ps(ya0, xa0));
        let a1 = _mm256_div_ps(_mm256_min_ps(ya1, xa1), _mm256_max_ps(ya1, xa1));
        let a2 = _mm256_div_ps(_mm256_min_ps(ya2, xa2), _mm256_max_ps(ya2, xa2));
        let a3 = _mm256_div_ps(_mm256_min_ps(ya3, xa3), _mm256_max_ps(ya3, xa3));
        let a4 = _mm256_div_ps(_mm256_min_ps(ya4, xa4), _mm256_max_ps(ya4, xa4));
        let a5 = _mm256_div_ps(_mm256_min_ps(ya5, xa5), _mm256_max_ps(ya5, xa5));
        let a6 = _mm256_div_ps(_mm256_min_ps(ya6, xa6), _mm256_max_ps(ya6, xa6));
        let a7 = _mm256_div_ps(_mm256_min_ps(ya7, xa7), _mm256_max_ps(ya7, xa7));

        // --- Polynomial (Horner) ---
        let sq0 = _mm256_mul_ps(a0, a0); let sq1 = _mm256_mul_ps(a1, a1);
        let sq2 = _mm256_mul_ps(a2, a2); let sq3 = _mm256_mul_ps(a3, a3);
        let sq4 = _mm256_mul_ps(a4, a4); let sq5 = _mm256_mul_ps(a5, a5);
        let sq6 = _mm256_mul_ps(a6, a6); let sq7 = _mm256_mul_ps(a7, a7);

        let p0 = _mm256_mul_ps(a0, _mm256_add_ps(c0, _mm256_mul_ps(sq0, _mm256_add_ps(c1, _mm256_mul_ps(sq0, _mm256_add_ps(c2, _mm256_mul_ps(sq0, c3)))))));
        let p1 = _mm256_mul_ps(a1, _mm256_add_ps(c0, _mm256_mul_ps(sq1, _mm256_add_ps(c1, _mm256_mul_ps(sq1, _mm256_add_ps(c2, _mm256_mul_ps(sq1, c3)))))));
        let p2 = _mm256_mul_ps(a2, _mm256_add_ps(c0, _mm256_mul_ps(sq2, _mm256_add_ps(c1, _mm256_mul_ps(sq2, _mm256_add_ps(c2, _mm256_mul_ps(sq2, c3)))))));
        let p3 = _mm256_mul_ps(a3, _mm256_add_ps(c0, _mm256_mul_ps(sq3, _mm256_add_ps(c1, _mm256_mul_ps(sq3, _mm256_add_ps(c2, _mm256_mul_ps(sq3, c3)))))));
        let p4 = _mm256_mul_ps(a4, _mm256_add_ps(c0, _mm256_mul_ps(sq4, _mm256_add_ps(c1, _mm256_mul_ps(sq4, _mm256_add_ps(c2, _mm256_mul_ps(sq4, c3)))))));
        let p5 = _mm256_mul_ps(a5, _mm256_add_ps(c0, _mm256_mul_ps(sq5, _mm256_add_ps(c1, _mm256_mul_ps(sq5, _mm256_add_ps(c2, _mm256_mul_ps(sq5, c3)))))));
        let p6 = _mm256_mul_ps(a6, _mm256_add_ps(c0, _mm256_mul_ps(sq6, _mm256_add_ps(c1, _mm256_mul_ps(sq6, _mm256_add_ps(c2, _mm256_mul_ps(sq6, c3)))))));
        let p7 = _mm256_mul_ps(a7, _mm256_add_ps(c0, _mm256_mul_ps(sq7, _mm256_add_ps(c1, _mm256_mul_ps(sq7, _mm256_add_ps(c2, _mm256_mul_ps(sq7, c3)))))));

        // --- Restore Quadrant ---
        let mut f0 = _mm256_blendv_ps(p0, _mm256_sub_ps(pi_2, p0), s_mask0);
        let mut f1 = _mm256_blendv_ps(p1, _mm256_sub_ps(pi_2, p1), s_mask1);
        let mut f2 = _mm256_blendv_ps(p2, _mm256_sub_ps(pi_2, p2), s_mask2);
        let mut f3 = _mm256_blendv_ps(p3, _mm256_sub_ps(pi_2, p3), s_mask3);
        let mut f4 = _mm256_blendv_ps(p4, _mm256_sub_ps(pi_2, p4), s_mask4);
        let mut f5 = _mm256_blendv_ps(p5, _mm256_sub_ps(pi_2, p5), s_mask5);
        let mut f6 = _mm256_blendv_ps(p6, _mm256_sub_ps(pi_2, p6), s_mask6);
        let mut f7 = _mm256_blendv_ps(p7, _mm256_sub_ps(pi_2, p7), s_mask7);

        f0 = _mm256_blendv_ps(f0, _mm256_sub_ps(pi, f0), _mm256_cmp_ps(x0, _mm256_setzero_ps(), _CMP_LT_OQ));
        f1 = _mm256_blendv_ps(f1, _mm256_sub_ps(pi, f1), _mm256_cmp_ps(x1, _mm256_setzero_ps(), _CMP_LT_OQ));
        f2 = _mm256_blendv_ps(f2, _mm256_sub_ps(pi, f2), _mm256_cmp_ps(x2, _mm256_setzero_ps(), _CMP_LT_OQ));
        f3 = _mm256_blendv_ps(f3, _mm256_sub_ps(pi, f3), _mm256_cmp_ps(x3, _mm256_setzero_ps(), _CMP_LT_OQ));
        f4 = _mm256_blendv_ps(f4, _mm256_sub_ps(pi, f4), _mm256_cmp_ps(x4, _mm256_setzero_ps(), _CMP_LT_OQ));
        f5 = _mm256_blendv_ps(f5, _mm256_sub_ps(pi, f5), _mm256_cmp_ps(x5, _mm256_setzero_ps(), _CMP_LT_OQ));
        f6 = _mm256_blendv_ps(f6, _mm256_sub_ps(pi, f6), _mm256_cmp_ps(x6, _mm256_setzero_ps(), _CMP_LT_OQ));
        f7 = _mm256_blendv_ps(f7, _mm256_sub_ps(pi, f7), _mm256_cmp_ps(x7, _mm256_setzero_ps(), _CMP_LT_OQ));

        f0 = _mm256_xor_ps(f0, _mm256_and_ps(y0, sign_bit));
        f1 = _mm256_xor_ps(f1, _mm256_and_ps(y1, sign_bit));
        f2 = _mm256_xor_ps(f2, _mm256_and_ps(y2, sign_bit));
        f3 = _mm256_xor_ps(f3, _mm256_and_ps(y3, sign_bit));
        f4 = _mm256_xor_ps(f4, _mm256_and_ps(y4, sign_bit));
        f5 = _mm256_xor_ps(f5, _mm256_and_ps(y5, sign_bit));
        f6 = _mm256_xor_ps(f6, _mm256_and_ps(y6, sign_bit));
        f7 = _mm256_xor_ps(f7, _mm256_and_ps(y7, sign_bit));

        // Non-Temporal Stores (VMOVNTPS)
        _mm256_stream_ps(res.as_ptr().add(i) as *mut f32, f0);
        _mm256_stream_ps(res.as_ptr().add(i + 8) as *mut f32, f1);
        _mm256_stream_ps(res.as_ptr().add(i + 16) as *mut f32, f2);
        _mm256_stream_ps(res.as_ptr().add(i + 24) as *mut f32, f3);
        _mm256_stream_ps(res.as_ptr().add(i + 32) as *mut f32, f4);
        _mm256_stream_ps(res.as_ptr().add(i + 40) as *mut f32, f5);
        _mm256_stream_ps(res.as_ptr().add(i + 48) as *mut f32, f6);
        _mm256_stream_ps(res.as_ptr().add(i + 56) as *mut f32, f7);

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
