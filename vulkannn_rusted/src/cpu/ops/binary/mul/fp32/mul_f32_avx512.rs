//! AVX-512 Implementation for FP32 Multiply.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Mechanical 512-bit widening of the AVX2 kernel — same op, 16 lanes.
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks
//! AVX-512. Compiles on x86_64; gated on avx512f in dispatch. Memory-bound (§8).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn mul_f32_avx512(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        _mm512_storeu_ps(res.as_mut_ptr().add(i), _mm512_mul_ps(va, vb));
    }
    for i in n16..n {
        res[i] = a[i] * b[i];
    }
}
