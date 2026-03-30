//! ARM Scalable Vector Extension 2 (SVE2) Implementation for FP32 Addition.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
pub unsafe fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // SVE2 ARMv9-A Optimization.
    // This kernel utilizes VL-agnostic loops with potential for non-temporal streaming 
    // hints available in advanced SVE2 implementations (e.g. Neoverse V1/V2).
    while i < n {
        let pg = svwhilelt_b32_u64(i as u64, n as u64);

        // Load
        let va = svld1_f32(pg, a.as_ptr().add(i));
        let vb = svld1_f32(pg, b.as_ptr().add(i));

        // Compute
        let vr = svadd_f32_z(pg, va, vb);

        // Store: Some SVE2 implementations benefit from specialized store patterns.
        svst1_f32(pg, res.as_ptr().add(i) as *mut f32, vr);

        i += svcntw();
    }
}
