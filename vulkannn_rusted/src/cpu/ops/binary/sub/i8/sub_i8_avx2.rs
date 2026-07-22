//! AVX2 Implementation for I8 Sub (saturating, `_mm256_subs_epi8`).
//! Transcribed from cpu_old/ops/binary/sub/sub_i8.rs (legacy avx2 kernel).
//!
//! BENCH: PENDING (hw: x86_64 w/ AVX2). Reference box (i5-3450) lacks AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sub_i8_avx2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n = a.len();
    let n32 = (n / 32) * 32;
    for i in (0..n32).step_by(32) {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, _mm256_subs_epi8(va, vb));
    }
    for i in n32..n {
        res[i] = a[i].saturating_sub(b[i]);
    }
}
