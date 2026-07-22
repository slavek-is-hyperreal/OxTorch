//! NEON Implementation for F16 Multiply (f32 upcast path).
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Transcribed from cpu_old/ops/binary/mul/mul_f16.rs (legacy neon kernel, incl.
//! the u16<->f16 reinterpret fix that legacy was missing).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.
//! Compiles under `cargo check --target aarch64-unknown-linux-gnu`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn mul_f16_neon(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let n = a.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let va = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a.as_ptr().add(i) as *const u16)));
        let vb = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b.as_ptr().add(i) as *const u16)));
        let vr = vmulq_f32(va, vb);
        vst1_u16(
            res.as_mut_ptr().add(i) as *mut u16,
            vreinterpret_u16_f16(vcvt_f16_f32(vr)),
        );
    }
    for i in n4..n {
        res[i] = half::f16::from_f32(a[i].to_f32() * b[i].to_f32());
    }
}
