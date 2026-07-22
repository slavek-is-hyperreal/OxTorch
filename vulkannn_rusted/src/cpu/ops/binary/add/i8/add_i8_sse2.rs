//! SSE2 (no-AVX tier) Implementation for I8 Add (saturating, `_mm_adds_epi8`).
//! Part of the OxTorch Scientific-Grade Specialization Matrix. Hardware
//! saturating byte-add — correct by construction (unlike legacy's u64 SWAR,
//! which leaked carries across byte lanes; see i8/mod.rs).
//!
//! BENCH: PENDING (needs an i8 bench harness — bench_binary_f32_variant is
//! f32-only). Measurable on this box; harness deferred with f16c to Wave 2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_i8_sse2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n = a.len();
    let n16 = (n / 16) * 16;
    for i in (0..n16).step_by(16) {
        let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, _mm_adds_epi8(va, vb));
    }
    for i in n16..n {
        res[i] = a[i].saturating_add(b[i]);
    }
}
