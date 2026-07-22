//! AVX-512 ReLU for FP32 (`_mm512_max_ps` with zero). Mechanical 512-bit widen.
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX-512F). Reference box (i5-3450) lacks AVX-512.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn relu(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = _mm512_setzero_ps();
    let n = in_buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let v = _mm512_loadu_ps(in_buf.as_ptr().add(i));
        _mm512_storeu_ps(out_buf.as_mut_ptr().add(i), _mm512_max_ps(v, zero));
    }
    for i in n16..n {
        out_buf[i] = in_buf[i].max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_inplace(buf: &mut [f32]) {
    let zero = _mm512_setzero_ps();
    let n = buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let ptr = buf.as_mut_ptr().add(i);
        _mm512_storeu_ps(ptr, _mm512_max_ps(_mm512_loadu_ps(ptr), zero));
    }
    for x in buf[n16..].iter_mut() {
        *x = x.max(0.0);
    }
}
