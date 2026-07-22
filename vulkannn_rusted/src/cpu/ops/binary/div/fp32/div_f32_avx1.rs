//! AVX1 Implementation for FP32 Divide.
//! Transcribed from cpu_old/ops/binary/div/div_f32.rs (legacy "avx" kernel).
//! SIMD body uses raw `_mm256_div_ps` (no /0 guard, matching legacy); guarded
//! scalar tail (see div_f32_scalar for the legacy /0 quirk).
//!
//! BENCH: 2.0–4.0x vs scalar (i5-3450, `cargo bench -- div_f32`, 2026-07):
//! 4K 4.0x, 64K 3.92x, 1M 2.05x. Plain (cached) stores — no NT penalty at
//! small N, so it beats scalar everywhere. ~parity with sse2 (both divide-bound).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn div_f32_avx1(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(res.as_mut_ptr().add(i), _mm256_div_ps(va, vb));
    }
    for i in n8..n {
        res[i] = if b[i] != 0.0 { a[i] / b[i] } else { 0.0 };
    }
}
