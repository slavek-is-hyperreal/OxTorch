//! ARM Scalable Vector Extension (SVE) Implementation for FP32 Addition.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
pub unsafe fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // SVE Agnostic Vector Loop. 
    // This kernel scales from 128-bit (A64FX / Graviton) to 2048-bit (Future HPC).
    while i < n {
        // Generate predicate for the remaining elements
        let pg = svptrue_b32(); // Assuming full vector processing, or use svwhilelt_b32_s32
        
        // svwhilelt_b32_u64 generates a predicate for a range of elements [i, n)
        let pg_dyn = svwhilelt_b32_u64(i as u64, n as u64);

        // Load
        let va = svld1_f32(pg_dyn, a.as_ptr().add(i));
        let vb = svld1_f32(pg_dyn, b.as_ptr().add(i));

        // Compute
        let vr = svadd_f32_z(pg_dyn, va, vb);

        // Store
        svst1_f32(pg_dyn, res.as_ptr().add(i) as *mut f32, vr);

        // Advance by the number of elements in the vector (VL)
        i += svcntw();
    }
}
