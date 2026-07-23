//! NEON gelu (tanh-approx) — reuses tanh NEON core.
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use crate::cpu::ops::unary::tanh::fp32::tanh_f32_neon::tanh4;

#[cfg(target_arch = "aarch64")]
unsafe fn gelu4(x: float32x4_t) -> float32x4_t {
    let k = vdupq_n_f32(0.7978845608); let c = vdupq_n_f32(0.044715);
    let half = vdupq_n_f32(0.5); let one = vdupq_n_f32(1.0);
    let x3 = vmulq_f32(vmulq_f32(x, x), x);
    let inner = vmulq_f32(k, vaddq_f32(x, vmulq_f32(c, x3)));
    vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh4(inner)))
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn gelu(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { vst1q_f32(out_buf.as_mut_ptr().add(i), gelu4(vld1q_f32(in_buf.as_ptr().add(i)))); }
    for i in n4..n { out_buf[i] = super::gelu_f32_scalar::gelu_one(in_buf[i]); }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn gelu_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { let p = buf.as_mut_ptr().add(i); vst1q_f32(p, gelu4(vld1q_f32(p))); }
    for x in buf[n4..].iter_mut() { *x = super::gelu_f32_scalar::gelu_one(*x); }
}
