//! NEON ReLU for I8 (`vmaxq_s8` with zero). Mechanical ARM analog (legacy had no
//! i8 neon relu; this is the obvious correct kernel, matches scalar semantics).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
    let zero = vdupq_n_s8(0);
    let n = in_buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let v = vld1q_s8(in_buf.as_ptr().add(i));
        vst1q_s8(out_buf.as_mut_ptr().add(i), vmaxq_s8(v, zero));
    }
    for i in n16..n {
        out_buf[i] = in_buf[i].max(0i8);
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn relu_inplace(buf: &mut [i8]) {
    let zero = vdupq_n_s8(0);
    let n = buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let ptr = buf.as_mut_ptr().add(i);
        vst1q_s8(ptr, vmaxq_s8(vld1q_s8(ptr), zero));
    }
    for x in buf[n16..].iter_mut() {
        *x = (*x).max(0i8);
    }
}
