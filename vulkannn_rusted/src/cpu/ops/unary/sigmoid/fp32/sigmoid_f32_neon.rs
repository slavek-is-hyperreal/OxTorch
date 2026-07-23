//! NEON sigmoid: reuses the exp NEON core; 1/(1+exp(-x)).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_neon::exp4;

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn sig4(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let neg_abs = vnegq_f32(vabsq_f32(x));
    let z = exp4(neg_abs);
    let mask = vcltq_f32(x, vdupq_n_f32(0.0));
    let num = vbslq_f32(mask, z, one);
    vdivq_f32(num, vaddq_f32(one, z))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn sigmoid(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        vst1q_f32(out_buf.as_mut_ptr().add(i), sig4(vld1q_f32(in_buf.as_ptr().add(i))));
    }
    for i in n4..n { out_buf[i] = super::sigmoid_f32_scalar::sigmoid_one(in_buf[i]); }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn sigmoid_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        vst1q_f32(ptr, sig4(vld1q_f32(ptr)));
    }
    for x in buf[n4..].iter_mut() { *x = super::sigmoid_f32_scalar::sigmoid_one(*x); }
}
