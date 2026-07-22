//! AVX2 Implementation for I8 Multiply (saturating, widen-to-i16 path).
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Transcribed VERBATIM from cpu_old/ops/binary/mul/mul_i8.rs (legacy avx2).
//! Saturation semantics live in `_mm256_packs_epi16`; do not "simplify".
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_i8_avx2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n = a.len();
    let n32 = (n / 32) * 32;
    for i in (0..n32).step_by(32) {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let lo_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let hi_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let lo_b = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let hi_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        let prod_lo = _mm256_mullo_epi16(lo_a, lo_b);
        let prod_hi = _mm256_mullo_epi16(hi_a, hi_b);

        let res_vec = _mm256_packs_epi16(prod_lo, prod_hi);
        let res_ordered = _mm256_permute4x64_epi64(res_vec, 0xD8);
        _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, res_ordered);
    }
    for i in n32..n {
        res[i] = a[i].saturating_mul(b[i]);
    }
}
