//! SSE2 (no-AVX tier) ReLU for FP32 (`_mm_max_ps` with zero).
//! Mechanical 128-bit narrowing of the AVX1 kernel.
//!
//! BENCH: PENDING (needs a unary bench harness — bench_binary_f32_variant is
//! binary-only). Memory-bound (§8); relu is a masked store, ~parity with scalar.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn relu(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = _mm_setzero_ps();
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = _mm_loadu_ps(in_buf.as_ptr().add(i));
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), _mm_max_ps(v, zero));
    }
    for i in n4..n {
        out_buf[i] = in_buf[i].max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn relu_inplace(buf: &mut [f32]) {
    let zero = _mm_setzero_ps();
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm_storeu_ps(ptr, _mm_max_ps(_mm_loadu_ps(ptr), zero));
    }
    for x in buf[n4..].iter_mut() {
        *x = x.max(0.0);
    }
}
