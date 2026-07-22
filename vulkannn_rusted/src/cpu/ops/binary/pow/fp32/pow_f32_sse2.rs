//! SSE2 (no-AVX tier) square kernel for the FP32 Pow exponent==2.0 fast path.
//! Mechanical 128-bit narrowing of the AVX1 square — `v*v`, 4 lanes.
//!
//! BENCH: PENDING (needs a unary-with-scalar bench harness). Square is
//! memory-bound (§8) — expect parity with auto-vectorised scalar, like mul.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn square(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = _mm_loadu_ps(in_buf.as_ptr().add(i));
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), _mm_mul_ps(v, v));
    }
    for i in n4..n {
        out_buf[i] = in_buf[i] * in_buf[i];
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn square_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = _mm_loadu_ps(ptr);
        _mm_storeu_ps(ptr, _mm_mul_ps(v, v));
    }
    for x in buf[n4..].iter_mut() {
        *x = *x * *x;
    }
}
