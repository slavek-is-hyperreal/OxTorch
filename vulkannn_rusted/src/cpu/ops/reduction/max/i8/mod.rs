//! I8 max — avx2 / sse4.1 / scalar. (`_mm*_max_epi8` is SSE4.1+, like relu_i8.)
//! NOTE: a GPR-only SWAR max tier (catalog A6) is NOT available — A6 depends on
//! the A8 per-lane compare mask, which was DEFERRED in SWAR Stage 1 (its first
//! formula failed the exhaustive gate; musl HASZERO is a detector, not a per-lane
//! mask). SWAR i8 max is a Stage-2 follow-up, blocked on a borrow-contained A8.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
pub fn max(buf: &[i8], initial: i8) -> i8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") { return unsafe { max_avx2(buf, initial) }; }
        if is_x86_feature_detected!("sse4.1") { return unsafe { max_sse41(buf, initial) }; }
    }
    buf.iter().fold(initial, |a, &b| a.max(b))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_avx2(buf: &[i8], initial: i8) -> i8 {
    let mut m = _mm256_set1_epi8(initial); let n32 = (buf.len()/32)*32;
    for i in (0..n32).step_by(32) { m = _mm256_max_epi8(m, _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i)); }
    let mut t=[0i8;32]; _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, m);
    let mut r = t.iter().fold(initial, |a,&b| a.max(b));
    for &x in &buf[n32..] { r = r.max(x); } r
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn max_sse41(buf: &[i8], initial: i8) -> i8 {
    let mut m = _mm_set1_epi8(initial); let n16 = (buf.len()/16)*16;
    for i in (0..n16).step_by(16) { m = _mm_max_epi8(m, _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i)); }
    let mut t=[0i8;16]; _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, m);
    let mut r = t.iter().fold(initial, |a,&b| a.max(b));
    for &x in &buf[n16..] { r = r.max(x); } r
}
