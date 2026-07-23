//! NEON silu: x/(1+exp(-x)), reusing the exp NEON core.
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_neon::exp4;

#[cfg(target_arch = "aarch64")]
unsafe fn silu4(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let e = exp4(vnegq_f32(x));
    vdivq_f32(x, vaddq_f32(one, e))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn silu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { vst1q_f32(out_buf.as_mut_ptr().add(i), silu4(vld1q_f32(in_buf.as_ptr().add(i)))); }
    for i in n4..n { out_buf[i] = super::silu_f32_scalar::silu_one(in_buf[i]); }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn silu_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { let p = buf.as_mut_ptr().add(i); vst1q_f32(p, silu4(vld1q_f32(p))); }
    for x in buf[n4..].iter_mut() { *x = super::silu_f32_scalar::silu_one(*x); }
}
