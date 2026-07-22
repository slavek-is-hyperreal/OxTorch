//! AVX1 Implementation for FP32 Multiply.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Transcribed from cpu_old/ops/binary/mul/mul_f32.rs (the legacy "avx" kernel).
//! Plain (cached) stores — memory-bound.
//!
//! BENCH: ~0.85–1.02x vs scalar (i5-3450, `cargo bench -- mul_f32`, 2026-07):
//! 4K 0.85x (256-bit setup overhead while in-cache), 64K 0.93x, 1M ~1.0x, prime
//! tail 1.06x. Bandwidth-bound (§8) — does not beat auto-vectorised scalar on
//! this single-channel-era box; kept as the AVX1 ISA tier for wider/faster HW.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn mul_f32_avx1(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(res.as_mut_ptr().add(i), _mm256_mul_ps(va, vb));
    }
    for i in n8..n {
        res[i] = a[i] * b[i];
    }
}
