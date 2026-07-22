//! SSE2 (no-AVX tier) Implementation for FP32 Multiply.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Mechanical 128-bit narrowing of the AVX1 kernel — same op, 4 lanes.
//!
//! BENCH: ~1.00x vs scalar (i5-3450, `cargo bench -- mul_f32`, 2026-07): 4K 1.02x,
//! 64K 1.01x, 1M 0.99x. Parity — mul is bandwidth-bound and rustc already
//! auto-vectorises the scalar loop to SSE, so this ties rather than wins (§8).
//! Kept as the explicit no-AVX ISA tier; expected to pull ahead only on CPUs
//! where memory outruns a single core's scalar throughput.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn mul_f32_sse2(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        _mm_storeu_ps(res.as_mut_ptr().add(i), _mm_mul_ps(va, vb));
    }
    for i in n4..n {
        res[i] = a[i] * b[i];
    }
}
