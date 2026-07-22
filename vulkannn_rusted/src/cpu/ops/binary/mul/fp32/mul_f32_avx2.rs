//! AVX2 Implementation for FP32 Multiply.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Transcribed from cpu_old/ops/binary/mul/mul_f32.rs (legacy avx2 kernel).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.
//! Compiles on x86_64. Memory-bound (§8) → expect near-parity with avx1.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], res: &mut [f32]) {
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
