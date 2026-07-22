//! NEON ReLU for FP32 (`vmaxq_f32` with zero).
//! Transcribed from cpu_old/ops/unary/relu/relu_f32.rs (legacy neon kernel).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn relu(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = vdupq_n_f32(0.0);
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(in_buf.as_ptr().add(i));
        vst1q_f32(out_buf.as_mut_ptr().add(i), vmaxq_f32(v, zero));
    }
    for i in n4..n {
        out_buf[i] = in_buf[i].max(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn relu_inplace(buf: &mut [f32]) {
    let zero = vdupq_n_f32(0.0);
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        vst1q_f32(ptr, vmaxq_f32(vld1q_f32(ptr), zero));
    }
    for x in buf[n4..].iter_mut() {
        *x = x.max(0.0);
    }
}
