//! SSE2 (no-AVX tier) negation for FP32 (`0 - x`). Mechanical narrowing of AVX1.
//!
//! BENCH: PENDING (needs a unary bench harness). Memory-bound (§8).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn neg(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = _mm_setzero_ps();
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = _mm_loadu_ps(in_buf.as_ptr().add(i));
        _mm_storeu_ps(out_buf.as_mut_ptr().add(i), _mm_sub_ps(zero, v));
    }
    for i in n4..n {
        out_buf[i] = -in_buf[i];
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn neg_inplace(buf: &mut [f32]) {
    let zero = _mm_setzero_ps();
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm_storeu_ps(ptr, _mm_sub_ps(zero, _mm_loadu_ps(ptr)));
    }
    for x in buf[n4..].iter_mut() {
        *x = -*x;
    }
}
