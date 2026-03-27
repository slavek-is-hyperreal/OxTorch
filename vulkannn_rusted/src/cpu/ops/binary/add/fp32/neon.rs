//! 128-bit NEON SIMD Implementation for FP32 Addition.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // ARM Cortex-A76 (Raspberry Pi 5) Optimization.
    // We unroll by 8 (8 * 4 = 32 floats per iteration) to saturate the 2x 256-bit buses 
    // and hide the 3-cycle FADD latency on the F1 pipe.
    while i + 31 < n {
        // Load Block A
        let a0 = vld1q_f32(a.as_ptr().add(i));
        let a1 = vld1q_f32(a.as_ptr().add(i + 4));
        let a2 = vld1q_f32(a.as_ptr().add(i + 8));
        let a3 = vld1q_f32(a.as_ptr().add(i + 12));
        let a4 = vld1q_f32(a.as_ptr().add(i + 16));
        let a5 = vld1q_f32(a.as_ptr().add(i + 20));
        let a6 = vld1q_f32(a.as_ptr().add(i + 24));
        let a7 = vld1q_f32(a.as_ptr().add(i + 28));

        // Load Block B
        let b0 = vld1q_f32(b.as_ptr().add(i));
        let b1 = vld1q_f32(b.as_ptr().add(i + 4));
        let b2 = vld1q_f32(b.as_ptr().add(i + 8));
        let b3 = vld1q_f32(b.as_ptr().add(i + 12));
        let b4 = vld1q_f32(b.as_ptr().add(i + 16));
        let b5 = vld1q_f32(b.as_ptr().add(i + 20));
        let b6 = vld1q_f32(b.as_ptr().add(i + 24));
        let b7 = vld1q_f32(b.as_ptr().add(i + 28));

        // Compute: ARMv8-A FADD
        let r0 = vaddq_f32(a0, b0);
        let r1 = vaddq_f32(a1, b1);
        let r2 = vaddq_f32(a2, b2);
        let r3 = vaddq_f32(a3, b3);
        let r4 = vaddq_f32(a4, b4);
        let r5 = vaddq_f32(a5, b5);
        let r6 = vaddq_f32(a6, b6);
        let r7 = vaddq_f32(a7, b7);

        // Store
        vst1q_f32(res.as_ptr().add(i) as *mut f32, r0);
        vst1q_f32(res.as_ptr().add(i + 4) as *mut f32, r1);
        vst1q_f32(res.as_ptr().add(i + 8) as *mut i8 as *mut f32, r2); // Note: casting used for alignment safety in some compilers
        vst1q_f32(res.as_ptr().add(i + 12) as *mut f32, r3);
        vst1q_f32(res.as_ptr().add(i + 16) as *mut f32, r4);
        vst1q_f32(res.as_ptr().add(i + 20) as *mut f32, r5);
        vst1q_f32(res.as_ptr().add(i + 24) as *mut f32, r6);
        vst1q_f32(res.as_ptr().add(i + 28) as *mut f32, r7);

        i += 32;
    }

    // Scalar remainder
    for remain in i..n {
        res[remain] = a[remain] + b[remain];
    }
}
