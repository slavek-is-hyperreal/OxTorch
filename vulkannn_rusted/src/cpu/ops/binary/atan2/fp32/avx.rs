//! 256-bit AVX SIMD Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Strictly branchless using bitmasking to saturate the execution units.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let sign_bit = _mm256_set1_ps(f32::from_bits(0x80000000));
    
    // Polynomial Constants
    let c0 = _mm256_set1_ps(0.99978784);
    let c1 = _mm256_set1_ps(-0.32580840);
    let c2 = _mm256_set1_ps(0.15557865);
    let c3 = _mm256_set1_ps(-0.04432655);
    
    let pi = _mm256_set1_ps(core::f32::consts::PI);
    let pi_2 = _mm256_set1_ps(core::f32::consts::FRAC_PI_2);

    // Unroll by 4 (4 * 8 = 32 floats per iteration)
    while i + 31 < n {
        let (y0, x0) = (_mm256_loadu_ps(y.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i)));
        let (y1, x1) = (_mm256_loadu_ps(y.as_ptr().add(i+8)), _mm256_loadu_ps(x.as_ptr().add(i+8)));
        let (y2, x2) = (_mm256_loadu_ps(y.as_ptr().add(i+16)), _mm256_loadu_ps(x.as_ptr().add(i+16)));
        let (y3, x3) = (_mm256_loadu_ps(y.as_ptr().add(i+24)), _mm256_loadu_ps(x.as_ptr().add(i+24)));

        let ya0 = _mm256_and_ps(y0, abs_mask);
        let xa0 = _mm256_and_ps(x0, abs_mask);
        let ya1 = _mm256_and_ps(y1, abs_mask);
        let xa1 = _mm256_and_ps(x1, abs_mask);
        let ya2 = _mm256_and_ps(y2, abs_mask);
        let xa2 = _mm256_and_ps(x2, abs_mask);
        let ya3 = _mm256_and_ps(y3, abs_mask);
        let xa3 = _mm256_and_ps(x3, abs_mask);

        let swap_mask0 = _mm256_cmp_ps(ya0, xa0, _CMP_GT_OQ);
        let swap_mask1 = _mm256_cmp_ps(ya1, xa1, _CMP_GT_OQ);
        let swap_mask2 = _mm256_cmp_ps(ya2, xa2, _CMP_GT_OQ);
        let swap_mask3 = _mm256_cmp_ps(ya3, xa3, _CMP_GT_OQ);

        let num0 = _mm256_min_ps(ya0, xa0);
        let den0 = _mm256_max_ps(ya0, xa0);
        let num1 = _mm256_min_ps(ya1, xa1);
        let den1 = _mm256_max_ps(ya1, xa1);
        let num2 = _mm256_min_ps(ya2, xa2);
        let den2 = _mm256_max_ps(ya2, xa2);
        let num3 = _mm256_min_ps(ya3, xa3);
        let den3 = _mm256_max_ps(ya3, xa3);

        // a = num / den (Division is pipelined on modern AVX but slow)
        let a0 = _mm256_div_ps(num0, den0);
        let a1 = _mm256_div_ps(num1, den1);
        let a2 = _mm256_div_ps(num2, den2);
        let a3 = _mm256_div_ps(num3, den3);

        let s0 = _mm256_mul_ps(a0, a0);
        let s1 = _mm256_mul_ps(a1, a1);
        let s2 = _mm256_mul_ps(a2, a2);
        let s3 = _mm256_mul_ps(a3, a3);

        // Polynomial (Horner)
        let mut p0 = _mm256_add_ps(c2, _mm256_mul_ps(s0, c3));
        let mut p1 = _mm256_add_ps(c2, _mm256_mul_ps(s1, c3));
        let mut p2 = _mm256_add_ps(c2, _mm256_mul_ps(s2, c3));
        let mut p3 = _mm256_add_ps(c2, _mm256_mul_ps(s3, c3));

        p0 = _mm256_add_ps(c1, _mm256_mul_ps(s0, p0));
        p1 = _mm256_add_ps(c1, _mm256_mul_ps(s1, p1));
        p2 = _mm256_add_ps(c1, _mm256_mul_ps(s2, p2));
        p3 = _mm256_add_ps(c1, _mm256_mul_ps(s3, p3));

        p0 = _mm256_add_ps(c0, _mm256_mul_ps(s0, p0));
        p1 = _mm256_add_ps(c0, _mm256_mul_ps(s1, p1));
        p2 = _mm256_add_ps(c0, _mm256_mul_ps(s2, p2));
        p3 = _mm256_add_ps(c0, _mm256_mul_ps(s3, p3));

        p0 = _mm256_mul_ps(a0, p0);
        p1 = _mm256_mul_ps(a1, p1);
        p2 = _mm256_mul_ps(a2, p2);
        p3 = _mm256_mul_ps(a3, p3);

        // --- Branchless Restorative logic ---
        // 1. Swap if ya > xa
        let p_swap0 = _mm256_sub_ps(pi_2, p0);
        let p_swap1 = _mm256_sub_ps(pi_2, p1);
        let p_swap2 = _mm256_sub_ps(pi_2, p2);
        let p_swap3 = _mm256_sub_ps(pi_2, p3);
        p0 = _mm256_blendv_ps(p0, p_swap0, swap_mask0);
        p1 = _mm256_blendv_ps(p1, p_swap1, swap_mask1);
        p2 = _mm256_blendv_ps(p2, p_swap2, swap_mask2);
        p3 = _mm256_blendv_ps(p3, p_swap3, swap_mask3);

        // 2. Mirror if x < 0
        let x_neg_mask0 = _mm256_cmp_ps(x0, _mm256_setzero_ps(), _CMP_LT_OQ);
        let x_neg_mask1 = _mm256_cmp_ps(x1, _mm256_setzero_ps(), _CMP_LT_OQ);
        let x_neg_mask2 = _mm256_cmp_ps(x2, _mm256_setzero_ps(), _CMP_LT_OQ);
        let x_neg_mask3 = _mm256_cmp_ps(x3, _mm256_setzero_ps(), _CMP_LT_OQ);
        let p_mir0 = _mm256_sub_ps(pi, p0);
        let p_mir1 = _mm256_sub_ps(pi, p1);
        let p_mir2 = _mm256_sub_ps(pi, p2);
        let p_mir3 = _mm256_sub_ps(pi, p3);
        p0 = _mm256_blendv_ps(p0, p_mir0, x_neg_mask0);
        p1 = _mm256_blendv_ps(p1, p_mir1, x_neg_mask1);
        p2 = _mm256_blendv_ps(p2, p_mir2, x_neg_mask2);
        p3 = _mm256_blendv_ps(p3, p_mir3, x_neg_mask3);

        // 3. Negate based on original Y sign
        let y_sign0 = _mm256_and_ps(y0, sign_bit);
        let y_sign1 = _mm256_and_ps(y1, sign_bit);
        let y_sign2 = _mm256_and_ps(y2, sign_bit);
        let y_sign3 = _mm256_and_ps(y3, sign_bit);
        p0 = _mm256_xor_ps(p0, y_sign0);
        p1 = _mm256_xor_ps(p1, y_sign1);
        p2 = _mm256_xor_ps(p2, y_sign2);
        p3 = _mm256_xor_ps(p3, y_sign3);

        // Store Non-Temporal
        _mm256_stream_ps(res.as_ptr().add(i) as *mut f32, p0);
        _mm256_stream_ps(res.as_ptr().add(i + 8) as *mut f32, p1);
        _mm256_stream_ps(res.as_ptr().add(i + 16) as *mut f32, p2);
        _mm256_stream_ps(res.as_ptr().add(i + 24) as *mut f32, p3);

        i += 32;
    }

    // Scalar Rest
    for remain in i..n {
        let (yi, xi) = (y[remain], x[remain]);
        let y_abs = yi.abs();
        let x_abs = xi.abs();
        let (a, swap) = if y_abs > x_abs { (x_abs/y_abs, true) } else { (y_abs/x_abs, false) };
        let s = a * a;
        let mut p = a * (0.99978784 + s * (-0.32580840 + s * (0.15557865 + s * -0.04432655)));
        if swap { p = core::f32::consts::FRAC_PI_2 - p; }
        if xi < 0.0 { p = core::f32::consts::PI - p; }
        if yi < 0.0 { p = -p; }
        res[remain] = p;
    }
    
    _mm_sfence();
}
