//! SSE2 (no-AVX tier) Implementation for FP32 Divide.
//! Mechanical 128-bit narrowing of the AVX1 kernel — same op, 4 lanes.
//! SIMD body uses a RAW divide (no /0 guard), matching legacy; the guarded
//! scalar formula is used only for the tail (see div_f32_scalar for the quirk).
//!
//! BENCH: 2.1–3.9x vs scalar (i5-3450, `cargo bench -- div_f32`, 2026-07):
//! 4K 3.9x, 64K 3.86x, 1M 2.1x. Unlike add/mul, divide is high-latency and
//! rustc does NOT auto-vectorise the scalar loop, so the no-AVX tier wins big.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn div_f32_sse2(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        _mm_storeu_ps(res.as_mut_ptr().add(i), _mm_div_ps(va, vb));
    }
    for i in n4..n {
        res[i] = if b[i] != 0.0 { a[i] / b[i] } else { 0.0 };
    }
}
