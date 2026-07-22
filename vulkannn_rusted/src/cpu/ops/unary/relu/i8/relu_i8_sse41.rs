//! SSE4.1 ReLU for I8 (`_mm_max_epi8` with zero) — the no-AVX2 x86 tier.
//! `_mm_max_epi8` is SSE4.1 (SSE2 has no signed byte max); every AVX-capable CPU
//! has SSE4.1, and the reference i5-3450 does too. Dispatched on runtime
//! `sse4.1` detection (not `active_arch`, which does not model SSE4.1).
//!
//! BENCH: PENDING (needs an i8 bench harness — deferred within Wave 2).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
    let zero = _mm_setzero_si128();
    let n = in_buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let v = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, _mm_max_epi8(v, zero));
    }
    for i in n16..n {
        out_buf[i] = in_buf[i].max(0i8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn relu_inplace(buf: &mut [i8]) {
    let zero = _mm_setzero_si128();
    let n = buf.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let ptr = buf.as_mut_ptr().add(i) as *mut __m128i;
        _mm_storeu_si128(ptr, _mm_max_epi8(_mm_loadu_si128(ptr), zero));
    }
    for x in buf[n16..].iter_mut() {
        *x = (*x).max(0i8);
    }
}
