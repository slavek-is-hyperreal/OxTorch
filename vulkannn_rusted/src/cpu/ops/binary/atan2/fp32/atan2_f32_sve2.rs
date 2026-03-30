//! Advanced Scalable Vector Extension 2 (SVE2) Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Maximizes ARM v9-A (Neoverse V2) throughput via deep register blocking.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    // Polynomial Constants (Minimax Remez)
    let c0 = svdup_n_f32(0.99978784);
    let c1 = svdup_n_f32(-0.32580840);
    let c2 = svdup_n_f32(0.15557865);
    let c3 = svdup_n_f32(-0.04432655);
    
    let pi = svdup_n_f32(core::f32::consts::PI);
    let pi_2 = svdup_n_f32(core::f32::consts::FRAC_PI_2);
    let zero = svdup_n_f32(0.0);

    // Advanced SVE2 Unrolling: 2 hardware-vector-length increments per iteration
    // This allows the out-of-order engine to overlap loads with slow VDIV/VMULL chains.
    while i + 2 * svcntw() <= n {
        let pg = svptrue_b32();

        // Load Block
        let y0 = svld1_f32(pg, y.as_ptr().add(i));
        let x0 = svld1_f32(pg, x.as_ptr().add(i));
        let y1 = svld1_f32(pg, y.as_ptr().add(i + svcntw()));
        let x1 = svld1_f32(pg, x.as_ptr().add(i + svcntw()));

        // --- Domain Reduction ---
        let (ya0, xa0) = (svabs_f32_x(pg, y0), svabs_f32_x(pg, x0));
        let (ya1, xa1) = (svabs_f32_x(pg, y1), svabs_f32_x(pg, x1));

        let m_swap0 = svcmpgt_f32(pg, ya0, xa0);
        let m_swap1 = svcmpgt_f32(pg, ya1, xa1);

        let a0 = svdiv_f32_x(pg, svmin_f32_x(pg, ya0, xa0), svmax_f32_x(pg, ya0, xa0));
        let a1 = svdiv_f32_x(pg, svmin_f32_x(pg, ya1, xa1), svmax_f32_x(pg, ya1, xa1));

        // --- Polynomial (Horner FMA) ---
        let sq0 = svmul_f32_x(pg, a0, a0); let sq1 = svmul_f32_x(pg, a1, a1);

        let mut p0 = svmla_f32_x(pg, c2, sq0, c3);
        let mut p1 = svmla_f32_x(pg, c2, sq1, c3);

        p0 = svmla_f32_x(pg, c1, sq0, p0);
        p1 = svmla_f32_x(pg, c1, sq1, p1);

        p0 = svmla_f32_x(pg, c0, sq0, p0);
        p1 = svmla_f32_x(pg, c0, sq1, p1);

        p0 = svmul_f32_x(pg, a0, p0);
        p1 = svmul_f32_x(pg, a1, p1);

        // --- Restore Quadrant ---
        p0 = svsel_f32(m_swap0, svsub_f32_x(pg, pi_2, p0), p0);
        p1 = svsel_f32(m_swap1, svsub_f32_x(pg, pi_2, p1), p1);

        p0 = svsel_f32(svcmplt_f32(pg, x0, zero), svsub_f32_x(pg, pi, p0), p0);
        p1 = svsel_f32(svcmplt_f32(pg, x1, zero), svsub_f32_x(pg, pi, p1), p1);

        let sign_mask = svreinterpret_f32_u32(svdup_n_u32(0x80000000));
        p0 = svreinterpret_f32_u32(sveor_u32_x(pg, svreinterpret_u32_f32(p0), svreinterpret_u32_f32(svand_f32_x(pg, y0, sign_mask))));
        p1 = svreinterpret_f32_u32(sveor_u32_x(pg, svreinterpret_u32_f32(p1), svreinterpret_u32_f32(svand_f32_x(pg, y1, sign_mask))));

        // Store
        svst1_f32(pg, res.as_mut_ptr().add(i), p0);
        svst1_f32(pg, res.as_mut_ptr().add(i + svcntw()), p1);

        i += 2 * svcntw();
    }

    // VLA Tail (Scalar or Single Predicated Vector)
    while i < n {
        let pg = svwhilelt_b32_u64(i as u64, n as u64);
        let y0 = svld1_f32(pg, y.as_ptr().add(i));
        let x0 = svld1_f32(pg, x.as_ptr().add(i));
        let ya = svabs_f32_x(pg, y0);
        let xa = svabs_f32_x(pg, x0);
        let m_swap = svcmpgt_f32(pg, ya, xa);
        let a = svdiv_f32_x(pg, svmin_f32_x(pg, ya, xa), svmax_f32_x(pg, ya, xa));
        let s = svmul_f32_x(pg, a, a);
        let mut p = svmla_f32_x(pg, c0, s, svmla_f32_x(pg, c1, s, svmla_f32_x(pg, c2, s, c3)));
        p = svmul_f32_x(pg, a, p);
        p = svsel_f32(m_swap, svsub_f32_x(pg, pi_2, p), p);
        p = svsel_f32(svcmplt_f32(pg, x0, zero), svsub_f32_x(pg, pi, p), p);
        let sign_mask = svreinterpret_f32_u32(svdup_n_u32(0x80000000));
        p = svreinterpret_f32_u32(sveor_u32_x(pg, svreinterpret_u32_f32(p), svreinterpret_u32_f32(svand_f32_x(pg, y0, sign_mask))));
        svst1_f32(pg, res.as_mut_ptr().add(i), p);
        i += svcntw();
    }
}
