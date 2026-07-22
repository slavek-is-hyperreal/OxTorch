//! NEON negation for FP32 (`vnegq_f32`). Transcribed from cpu_old neg_f32.
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn neg(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(in_buf.as_ptr().add(i));
        vst1q_f32(out_buf.as_mut_ptr().add(i), vnegq_f32(v));
    }
    for i in n4..n {
        out_buf[i] = -in_buf[i];
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn neg_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        vst1q_f32(ptr, vnegq_f32(vld1q_f32(ptr)));
    }
    for x in buf[n4..].iter_mut() {
        *x = -*x;
    }
}
