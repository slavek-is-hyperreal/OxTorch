//! F16C Implementation for F16 Divide (hardware f16<->f32 convert + AVX div).
//! Transcribed from cpu_old/ops/binary/div/div_f16.rs (legacy f16c kernel).
//! SIMD body raw divide; guarded scalar tail (/0 -> 0.0 legacy quirk).
//!
//! BENCH: PENDING (needs an f16 bench harness — lands in Wave 2 with the f16
//! transcendentals). Measurable on this box (i5-3450 has F16C).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn div_f16_f16c(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let n = a.len();
    let n8 = (n / 8) * 8;
    for i in (0..n8).step_by(8) {
        let va = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        let vr = _mm256_div_ps(va, vb);
        _mm_storeu_si128(
            res.as_mut_ptr().add(i) as *mut __m128i,
            _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT),
        );
    }
    for i in n8..n {
        let vb = b[i].to_f32();
        res[i] = half::f16::from_f32(if vb != 0.0 { a[i].to_f32() / vb } else { 0.0 });
    }
}
