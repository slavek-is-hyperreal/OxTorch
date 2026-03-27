//! Scalable Vector Extension (SVE) Implementation for FP32 Atan2.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Vector-Length Agnostic (VLA) design with predicated quadrant logic.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
pub unsafe fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    let n = y.len();
    let mut i = 0;

    // SVE Predicate Initialization (all elements enabled for constants)
    let pg_all = svptrue_b32();
    
    // Polynomial Constants (Minimax Remez)
    let c0 = svdup_n_f32(0.99978784);
    let c1 = svdup_n_f32(-0.32580840);
    let c2 = svdup_n_f32(0.15557865);
    let c3 = svdup_n_f32(-0.04432655);
    
    let pi = svdup_n_f32(core::f32::consts::PI);
    let pi_2 = svdup_n_f32(core::f32::consts::FRAC_PI_2);
    let zero = svdup_n_f32(0.0);

    // Vector-Length Agnostic Loop
    while i < n {
        // Generate predicate for remaining elements
        let pg = svwhilelt_b32_u64(i as u64, n as u64);

        // Load Tensors (Predicated)
        let y_vec = svld1_f32(pg, y.as_ptr().add(i));
        let x_vec = svld1_f32(pg, x.as_ptr().add(i));

        // --- Domain Reduction ---
        let y_abs = svabs_f32_x(pg, y_vec);
        let x_abs = svabs_f32_x(pg, x_vec);

        // Swap mask: ya > xa
        let m_swap = svcmpgt_f32(pg, y_abs, x_abs);

        // a = min / max
        let num = svmin_f32_x(pg, y_abs, x_abs);
        let den = svmax_f32_x(pg, y_abs, x_abs);
        let a = svdiv_f32_x(pg, num, den);

        // --- Polynomial (Horner FMA) ---
        let s = svmul_f32_x(pg, a, a);
        
        // p = c0 + s*(c1 + s*(c2 + s*c3))
        let mut p = svmla_f32_x(pg, c2, s, c3);
        p = svmla_f32_x(pg, c1, s, p);
        p = svmla_f32_x(pg, c0, s, p);
        p = svmul_f32_x(pg, a, p);

        // --- Restore Quadrant (SVE Selection) ---
        // 1. Swap if ya > xa
        p = svsel_f32(m_swap, svsub_f32_x(pg, pi_2, p), p);

        // 2. Mirror if x < 0
        let m_xneg = svcmplt_f32(pg, x_vec, zero);
        p = svsel_f32(m_xneg, svsub_f32_x(pg, pi, p), p);

        // 3. Negate based on original Y sign
        // Bitwise logic on SVE: svand_f32, sveor_f32
        let sign_mask = svreinterpret_f32_u32(svdup_n_u32(0x80000000));
        let y_sign = svand_f32_x(pg, y_vec, sign_mask);
        p = svreinterpret_f32_u32(sveor_u32_x(pg, svreinterpret_u32_f32(p), svreinterpret_u32_f32(y_sign)));

        // Store (Predicated)
        svst1_f32(pg, res.as_mut_ptr().add(i), p);

        // Increment by hardware vector length (e.g. 512-bit / 32-bit = 16 elements)
        i += svcntw();
    }
}
