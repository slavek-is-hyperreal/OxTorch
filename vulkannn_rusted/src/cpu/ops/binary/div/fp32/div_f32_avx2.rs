//! AVX2 Implementation for FP32 Divide.
//! Transcribed from cpu_old/ops/binary/div/div_f32.rs (legacy avx2 kernel).
//! Raw SIMD divide; guarded scalar tail (see div_f32_scalar for the /0 quirk).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn div_f32_avx2(a: &[f32], b: &[f32], res: &mut [f32]) {
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
