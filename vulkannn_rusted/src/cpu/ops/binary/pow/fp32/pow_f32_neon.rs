//! NEON square kernel for the FP32 Pow exponent==2.0 fast path.
//! Transcribed from cpu_old/ops/unary/pow/pow_f32.rs (legacy pow2_f32_neon).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn square(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(in_buf.as_ptr().add(i));
        vst1q_f32(out_buf.as_mut_ptr().add(i), vmulq_f32(v, v));
    }
    for i in n4..n {
        out_buf[i] = in_buf[i] * in_buf[i];
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn square_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = vld1q_f32(ptr);
        vst1q_f32(ptr, vmulq_f32(v, v));
    }
    for x in buf[n4..].iter_mut() {
        *x = *x * *x;
    }
}
