//! NEON Implementation for I8 Multiply (saturating, widen-to-i16 path).
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Transcribed VERBATIM from cpu_old/ops/binary/mul/mul_i8.rs (legacy neon).
//! Saturation lives in `vqmovn_s16`.
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn mul_i8_neon(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n = a.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));

        let lo_a = vmovl_s8(vget_low_s8(va));
        let hi_a = vmovl_s8(vget_high_s8(va));
        let lo_b = vmovl_s8(vget_low_s8(vb));
        let hi_b = vmovl_s8(vget_high_s8(vb));

        let prod_lo = vmulq_s16(lo_a, lo_b);
        let prod_hi = vmulq_s16(hi_a, hi_b);

        vst1q_s8(
            res.as_mut_ptr().add(i),
            vcombine_s8(vqmovn_s16(prod_lo), vqmovn_s16(prod_hi)),
        );
    }
    for i in n16..n {
        res[i] = a[i].saturating_mul(b[i]);
    }
}
