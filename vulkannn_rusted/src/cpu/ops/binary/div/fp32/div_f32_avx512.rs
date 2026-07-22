//! AVX-512 Implementation for FP32 Divide.
//! Mechanical 512-bit widening of the AVX2 kernel — same op, 16 lanes.
//! Raw SIMD divide; guarded scalar tail (see div_f32_scalar for the /0 quirk).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn div_f32_avx512(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        _mm512_storeu_ps(res.as_mut_ptr().add(i), _mm512_div_ps(va, vb));
    }
    for i in n16..n {
        res[i] = if b[i] != 0.0 { a[i] / b[i] } else { 0.0 };
    }
}
