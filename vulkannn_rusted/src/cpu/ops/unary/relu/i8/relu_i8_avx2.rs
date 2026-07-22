//! AVX2 ReLU for I8 (`_mm256_max_epi8` with zero).
//! Transcribed from cpu_old/ops/unary/relu/relu_i8.rs (legacy avx2 kernel).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
    let zero = _mm256_setzero_si256();
    let n = in_buf.len();
    let n32 = (n / 32) * 32;
    for i in (0..n32).step_by(32) {
        let v = _mm256_loadu_si256(in_buf.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(out_buf.as_mut_ptr().add(i) as *mut __m256i, _mm256_max_epi8(v, zero));
    }
    for i in n32..n {
        out_buf[i] = in_buf[i].max(0i8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_inplace(buf: &mut [i8]) {
    let zero = _mm256_setzero_si256();
    let n = buf.len();
    let n32 = (n / 32) * 32;
    for i in (0..n32).step_by(32) {
        let ptr = buf.as_mut_ptr().add(i) as *mut __m256i;
        _mm256_storeu_si256(ptr, _mm256_max_epi8(_mm256_loadu_si256(ptr), zero));
    }
    for x in buf[n32..].iter_mut() {
        *x = (*x).max(0i8);
    }
}
