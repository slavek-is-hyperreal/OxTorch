//! NEON Implementation for I8 Add (saturating, `vqaddq_s8`).
//! Transcribed from cpu_old/ops/binary/add/add_i8.rs (legacy neon kernel).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn add_i8_neon(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n = a.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        vst1q_s8(res.as_mut_ptr().add(i), vqaddq_s8(va, vb));
    }
    for i in n16..n {
        res[i] = a[i].saturating_add(b[i]);
    }
}
