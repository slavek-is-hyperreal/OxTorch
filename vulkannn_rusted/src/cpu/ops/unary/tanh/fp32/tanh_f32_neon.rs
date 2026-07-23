//! NEON tanh — Cephes two-branch, vbslq select.
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use crate::cpu::ops::unary::exp::fp32::exp_f32_neon::exp4;

#[cfg(target_arch = "aarch64")]
unsafe fn tanh4(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let ax = vabsq_f32(x);

    let s = exp4(vaddq_f32(ax, ax));
    let big_mag = vsubq_f32(one, vdivq_f32(two, vaddq_f32(s, one)));
    let signbits = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x8000_0000)));
    let big = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(big_mag), vreinterpretq_u32_f32(signbits)));

    let z = vmulq_f32(x, x);
    let mut p = vdupq_n_f32(-5.70498872745E-3);
    p = vaddq_f32(vmulq_f32(p, z), vdupq_n_f32(2.06390887954E-2));
    p = vaddq_f32(vmulq_f32(p, z), vdupq_n_f32(-5.37397155531E-2));
    p = vaddq_f32(vmulq_f32(p, z), vdupq_n_f32(1.33314422036E-1));
    p = vaddq_f32(vmulq_f32(p, z), vdupq_n_f32(-3.33332819422E-1));
    let small = vaddq_f32(vmulq_f32(vmulq_f32(p, z), x), x);

    let m = vcltq_f32(ax, vdupq_n_f32(0.625));
    vbslq_f32(m, small, big)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn tanh(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { vst1q_f32(out_buf.as_mut_ptr().add(i), tanh4(vld1q_f32(in_buf.as_ptr().add(i)))); }
    for i in n4..n { out_buf[i] = in_buf[i].tanh(); }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn tanh_inplace(buf: &mut [f32]) {
    let n = buf.len(); let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) { let p = buf.as_mut_ptr().add(i); vst1q_f32(p, tanh4(vld1q_f32(p))); }
    for x in buf[n4..].iter_mut() { *x = x.tanh(); }
}
