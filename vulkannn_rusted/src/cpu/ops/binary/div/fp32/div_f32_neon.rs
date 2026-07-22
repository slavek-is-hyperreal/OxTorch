//! NEON Implementation for FP32 Divide (ARMv8 vdivq_f32).
//! Transcribed from cpu_old/ops/binary/div/div_f32.rs (legacy neon kernel).
//! Raw SIMD divide; guarded scalar tail (see div_f32_scalar for the /0 quirk).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.
//! Compiles under `cargo check --target aarch64-unknown-linux-gnu`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn div_f32_neon(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        vst1q_f32(res.as_mut_ptr().add(i), vdivq_f32(va, vb));
    }
    for i in n4..n {
        res[i] = if b[i] != 0.0 { a[i] / b[i] } else { 0.0 };
    }
}
