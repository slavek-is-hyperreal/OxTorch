//! 128-bit NEON SIMD Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Optimized for ARM v8/v9 (Graviton/Apple Silicon) with branchless VBSL selection.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    // Polynomial Constants (Minimax Remez)
    let c0 = vdupq_n_f32(0.99978784);
    let c1 = vdupq_n_f32(-0.32580840);
    let c2 = vdupq_n_f32(0.15557865);
    let c3 = vdupq_n_f32(-0.04432655);
    
    let pi = vdupq_n_f32(core::f32::consts::PI);
    let pi_2 = vdupq_n_f32(core::f32::consts::FRAC_PI_2);
    let zero = vdupq_n_f32(0.0);

    // NEON Unrolling: 8 vectors (32 floats) per iteration
    // Maximizes throughput on superscalar ARM cores by hiding FMA/Div latencies.
    while i + 31 < n {
        // Load Block
        let (y0, x0) = (vld1q_f32(y.as_ptr().add(i)), vld1q_f32(x.as_ptr().add(i)));
        let (y1, x1) = (vld1q_f32(y.as_ptr().add(i + 4)), vld1q_f32(x.as_ptr().add(i + 4)));
        let (y2, x2) = (vld1q_f32(y.as_ptr().add(i + 8)), vld1q_f32(x.as_ptr().add(i + 8)));
        let (y3, x3) = (vld1q_f32(y.as_ptr().add(i + 12)), vld1q_f32(x.as_ptr().add(i + 12)));
        let (y4, x4) = (vld1q_f32(y.as_ptr().add(i + 16)), vld1q_f32(x.as_ptr().add(i + 16)));
        let (y5, x5) = (vld1q_f32(y.as_ptr().add(i + 20)), vld1q_f32(x.as_ptr().add(i + 20)));
        let (y6, x6) = (vld1q_f32(y.as_ptr().add(i + 24)), vld1q_f32(x.as_ptr().add(i + 24)));
        let (y7, x7) = (vld1q_f32(y.as_ptr().add(i + 28)), vld1q_f32(x.as_ptr().add(i + 28)));

        // --- Domain Reduction ---
        let (ya0, xa0) = (vabsq_f32(y0), vabsq_f32(x0));
        let (ya1, xa1) = (vabsq_f32(y1), vabsq_f32(x1));
        let (ya2, xa2) = (vabsq_f32(y2), vabsq_f32(x2));
        let (ya3, xa3) = (vabsq_f32(y3), vabsq_f32(x3));
        let (ya4, xa4) = (vabsq_f32(y4), vabsq_f32(x4));
        let (ya5, xa5) = (vabsq_f32(y5), vabsq_f32(x5));
        let (ya6, xa6) = (vabsq_f32(y6), vabsq_f32(x6));
        let (ya7, xa7) = (vabsq_f32(y7), vabsq_f32(x7));

        let s_mask0 = vcgtq_f32(ya0, xa0);
        let s_mask1 = vcgtq_f32(ya1, xa1);
        let s_mask2 = vcgtq_f32(ya2, xa2);
        let s_mask3 = vcgtq_f32(ya3, xa3);
        let s_mask4 = vcgtq_f32(ya4, xa4);
        let s_mask5 = vcgtq_f32(ya5, xa5);
        let s_mask6 = vcgtq_f32(ya6, xa6);
        let s_mask7 = vcgtq_f32(ya7, xa7);

        // a = min / max (Note: vdivq is relatively slow on NEON)
        let a0 = vdivq_f32(vminq_f32(ya0, xa0), vmaxq_f32(ya0, xa0));
        let a1 = vdivq_f32(vminq_f32(ya1, xa1), vmaxq_f32(ya1, xa1));
        let a2 = vdivq_f32(vminq_f32(ya2, xa2), vmaxq_f32(ya2, xa2));
        let a3 = vdivq_f32(vminq_f32(ya3, xa3), vmaxq_f32(ya3, xa3));
        let a4 = vdivq_f32(vminq_f32(ya4, xa4), vmaxq_f32(ya4, xa4));
        let a5 = vdivq_f32(vminq_f32(ya5, xa5), vmaxq_f32(ya5, xa5));
        let a6 = vdivq_f32(vminq_f32(ya6, xa6), vmaxq_f32(ya6, xa6));
        let a7 = vdivq_f32(vminq_f32(ya7, xa7), vmaxq_f32(ya7, xa7));

        // --- Polynomial (Horner using NEON FMA) ---
        let sq0 = vmulq_f32(a0, a0); let sq1 = vmulq_f32(a1, a1);
        let sq2 = vmulq_f32(a2, a2); let sq3 = vmulq_f32(a3, a3);
        let sq4 = vmulq_f32(a4, a4); let sq5 = vmulq_f32(a5, a5);
        let sq6 = vmulq_f32(a6, a6); let sq7 = vmulq_f32(a7, a7);

        // p = c0 + s*(c1 + s*(c2 + s*c3))
        let mut p0 = vfmaq_f32(c2, sq0, c3);
        let mut p1 = vfmaq_f32(c2, sq1, c3);
        let mut p2 = vfmaq_f32(c2, sq2, c3);
        let mut p3 = vfmaq_f32(c2, sq3, c3);
        let mut p4 = vfmaq_f32(c2, sq4, c3);
        let mut p5 = vfmaq_f32(c2, sq5, c3);
        let mut p6 = vfmaq_f32(c2, sq6, c3);
        let mut p7 = vfmaq_f32(c2, sq7, c3);

        p0 = vfmaq_f32(c1, sq0, p0);
        p1 = vfmaq_f32(c1, sq1, p1);
        p2 = vfmaq_f32(c1, sq2, p2);
        p3 = vfmaq_f32(c1, sq3, p3);
        p4 = vfmaq_f32(c1, sq4, p4);
        p5 = vfmaq_f32(c1, sq5, p5);
        p6 = vfmaq_f32(c1, sq6, p6);
        p7 = vfmaq_f32(c1, sq7, p7);

        p0 = vfmaq_f32(c0, sq0, p0);
        p1 = vfmaq_f32(c0, sq1, p1);
        p2 = vfmaq_f32(c0, sq2, p2);
        p3 = vfmaq_f32(c0, sq3, p3);
        p4 = vfmaq_f32(c0, sq4, p4);
        p5 = vfmaq_f32(c0, sq5, p5);
        p6 = vfmaq_f32(c0, sq6, p6);
        p7 = vfmaq_f32(c0, sq7, p7);

        p0 = vmulq_f32(a0, p0);
        p1 = vmulq_f32(a1, p1);
        p2 = vmulq_f32(a2, p2);
        p3 = vmulq_f32(a3, p3);
        p4 = vmulq_f32(a4, p4);
        p5 = vmulq_f32(a5, p5);
        p6 = vmulq_f32(a6, p6);
        p7 = vmulq_f32(a7, p7);

        // --- Restore Quadrant (VBSL Selection) ---
        // 1. Swap if ya > xa
        let mut f0 = vbslq_f32(s_mask0, vsubq_f32(pi_2, p0), p0);
        let mut f1 = vbslq_f32(s_mask1, vsubq_f32(pi_2, p1), p1);
        let mut f2 = vbslq_f32(s_mask2, vsubq_f32(pi_2, p2), p2);
        let mut f3 = vbslq_f32(s_mask3, vsubq_f32(pi_2, p3), p3);
        let mut f4 = vbslq_f32(s_mask4, vsubq_f32(pi_2, p4), p4);
        let mut f5 = vbslq_f32(s_mask5, vsubq_f32(pi_2, p5), p5);
        let mut f6 = vbslq_f32(s_mask6, vsubq_f32(pi_2, p6), p6);
        let mut f7 = vbslq_f32(s_mask7, vsubq_f32(pi_2, p7), p7);

        // 2. Mirror if x < 0
        f0 = vbslq_f32(vcltq_f32(x0, zero), vsubq_f32(pi, f0), f0);
        f1 = vbslq_f32(vcltq_f32(x1, zero), vsubq_f32(pi, f1), f1);
        f2 = vbslq_f32(vcltq_f32(x2, zero), vsubq_f32(pi, f2), f2);
        f3 = vbslq_f32(vcltq_f32(x3, zero), vsubq_f32(pi, f3), f3);
        f4 = vbslq_f32(vcltq_f32(x4, zero), vsubq_f32(pi, f4), f4);
        f5 = vbslq_f32(vcltq_f32(x5, zero), vsubq_f32(pi, f5), f5);
        f6 = vbslq_f32(vcltq_f32(x6, zero), vsubq_f32(pi, f6), f6);
        f7 = vbslq_f32(vcltq_f32(x7, zero), vsubq_f32(pi, f7), f7);

        // 3. Negate based on original Y sign
        // On NEON f32 sign-flip, XOR bit 31 is efficient via vnegq_f32 ?
        // Or simply vbslq against the sign bit of Y.
        // The report suggests XOR bits to avoid branching.
        // We'll use the bitwise XOR logic since it's consistent across architectures.
        let sign_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x80000000));
        f0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f0), vreinterpretq_u32_f32(vandq_f32(y0, sign_mask))));
        f1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f1), vreinterpretq_u32_f32(vandq_f32(y1, sign_mask))));
        f2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f2), vreinterpretq_u32_f32(vandq_f32(y2, sign_mask))));
        f3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f3), vreinterpretq_u32_f32(vandq_f32(y3, sign_mask))));
        f4 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f4), vreinterpretq_u32_f32(vandq_f32(y4, sign_mask))));
        f5 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f5), vreinterpretq_u32_f32(vandq_f32(y5, sign_mask))));
        f6 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f6), vreinterpretq_u32_f32(vandq_f32(y6, sign_mask))));
        f7 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f7), vreinterpretq_u32_f32(vandq_f32(y7, sign_mask))));

        // Store Block
        vst1q_f32(res.as_ptr().add(i), f0);
        vst1q_f32(res.as_ptr().add(i + 4), f1);
        vst1q_f32(res.as_ptr().add(i + 8), f2);
        vst1q_f32(res.as_ptr().add(i + 12), f3);
        vst1q_f32(res.as_ptr().add(i + 16), f4);
        vst1q_f32(res.as_ptr().add(i + 20), f5);
        vst1q_f32(res.as_ptr().add(i + 24), f6);
        vst1q_f32(res.as_ptr().add(i + 28), f7);

        i += 32;
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
}
