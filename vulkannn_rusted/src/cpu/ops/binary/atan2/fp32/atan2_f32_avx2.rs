//! 256-bit AVX2 + FMA Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Utilizes Fused Multiply-Add (FMA) and 8-vector register blocking to saturate the pipeline.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let sign_bit = _mm256_set1_ps(f32::from_bits(0x80000000));
    
    // Polynomial Constants (Minimax Remez)
    let c0 = _mm256_set1_ps(0.99978784);
    let c1 = _mm256_set1_ps(-0.32580840);
    let c2 = _mm256_set1_ps(0.15557865);
    let c3 = _mm256_set1_ps(-0.04432655);
    
    let pi = _mm256_set1_ps(core::f32::consts::PI);
    let pi_2 = _mm256_set1_ps(core::f32::consts::FRAC_PI_2);

    // AVX2 Register Blocking: 8 vectors (64 floats) per iteration
    // This hides the 4-cycle FMA latency by interleaving independent chains.
    while i + 63 < n {
        // Load Block
        let (y0, x0) = (_mm256_loadu_ps(y.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i)));
        let (y1, x1) = (_mm256_loadu_ps(y.as_ptr().add(i + 8)), _mm256_loadu_ps(x.as_ptr().add(i + 8)));
        let (y2, x2) = (_mm256_loadu_ps(y.as_ptr().add(i + 16)), _mm256_loadu_ps(x.as_ptr().add(i + 16)));
        let (y3, x3) = (_mm256_loadu_ps(y.as_ptr().add(i + 24)), _mm256_loadu_ps(x.as_ptr().add(i + 24)));
        let (y4, x4) = (_mm256_loadu_ps(y.as_ptr().add(i + 32)), _mm256_loadu_ps(x.as_ptr().add(i + 32)));
        let (y5, x5) = (_mm256_loadu_ps(y.as_ptr().add(i + 40)), _mm256_loadu_ps(x.as_ptr().add(i + 40)));
        let (y6, x6) = (_mm256_loadu_ps(y.as_ptr().add(i + 48)), _mm256_loadu_ps(x.as_ptr().add(i + 48)));
        let (y7, x7) = (_mm256_loadu_ps(y.as_ptr().add(i + 56)), _mm256_loadu_ps(x.as_ptr().add(i + 56)));

        // --- Domain Reduction ---
        let (ya0, xa0) = (_mm256_and_ps(y0, abs_mask), _mm256_and_ps(x0, abs_mask));
        let (ya1, xa1) = (_mm256_and_ps(y1, abs_mask), _mm256_and_ps(x1, abs_mask));
        let (ya2, xa2) = (_mm256_and_ps(y2, abs_mask), _mm256_and_ps(x2, abs_mask));
        let (ya3, xa3) = (_mm256_and_ps(y3, abs_mask), _mm256_and_ps(x3, abs_mask));
        let (ya4, xa4) = (_mm256_and_ps(y4, abs_mask), _mm256_and_ps(x4, abs_mask));
        let (ya5, xa5) = (_mm256_and_ps(y5, abs_mask), _mm256_and_ps(x5, abs_mask));
        let (ya6, xa6) = (_mm256_and_ps(y6, abs_mask), _mm256_and_ps(x6, abs_mask));
        let (ya7, xa7) = (_mm256_and_ps(y7, abs_mask), _mm256_and_ps(x7, abs_mask));

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

        // --- Polynomial (Horner using FMA) ---
        let sq0 = _mm256_mul_ps(a0, a0); let sq1 = _mm256_mul_ps(a1, a1);
        let sq2 = _mm256_mul_ps(a2, a2); let sq3 = _mm256_mul_ps(a3, a3);
        let sq4 = _mm256_mul_ps(a4, a4); let sq5 = _mm256_mul_ps(a5, a5);
        let sq6 = _mm256_mul_ps(a6, a6); let sq7 = _mm256_mul_ps(a7, a7);

        let mut p0 = _mm256_fmadd_ps(sq0, c3, c2);
        let mut p1 = _mm256_fmadd_ps(sq1, c3, c2);
        let mut p2 = _mm256_fmadd_ps(sq2, c3, c2);
        let mut p3 = _mm256_fmadd_ps(sq3, c3, c2);
        let mut p4 = _mm256_fmadd_ps(sq4, c3, c2);
        let mut p5 = _mm256_fmadd_ps(sq5, c3, c2);
        let mut p6 = _mm256_fmadd_ps(sq6, c3, c2);
        let mut p7 = _mm256_fmadd_ps(sq7, c3, c2);

        p0 = _mm256_fmadd_ps(sq0, p0, c1);
        p1 = _mm256_fmadd_ps(sq1, p1, c1);
        p2 = _mm256_fmadd_ps(sq2, p2, c1);
        p3 = _mm256_fmadd_ps(sq3, p3, c1);
        p4 = _mm256_fmadd_ps(sq4, p4, c1);
        p5 = _mm256_fmadd_ps(sq5, p5, c1);
        p6 = _mm256_fmadd_ps(sq6, p6, c1);
        p7 = _mm256_fmadd_ps(sq7, p7, c1);

        p0 = _mm256_fmadd_ps(sq0, p0, c0);
        p1 = _mm256_fmadd_ps(sq1, p1, c0);
        p2 = _mm256_fmadd_ps(sq2, p2, c0);
        p3 = _mm256_fmadd_ps(sq3, p3, c0);
        p4 = _mm256_fmadd_ps(sq4, p4, c0);
        p5 = _mm256_fmadd_ps(sq5, p5, c0);
        p6 = _mm256_fmadd_ps(sq6, p6, c0);
        p7 = _mm256_fmadd_ps(sq7, p7, c0);

        p0 = _mm256_mul_ps(a0, p0);
        p1 = _mm256_mul_ps(a1, p1);
        p2 = _mm256_mul_ps(a2, p2);
        p3 = _mm256_mul_ps(a3, p3);
        p4 = _mm256_mul_ps(a4, p4);
        p5 = _mm256_mul_ps(a5, p5);
        p6 = _mm256_mul_ps(a6, p6);
        p7 = _mm256_mul_ps(a7, p7);

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

        // Non-Temporal Stores
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
