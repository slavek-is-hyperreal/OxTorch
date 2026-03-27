#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Max reduction for I8 tensors.
pub fn max_i8(buf: &[i8], initial: i8) -> i8 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(128_000).map(|c| max_i8_serial(c, initial)).reduce(|| initial, |a, b| a.max(b));
    }
    max_i8_serial(buf, initial)
}

fn max_i8_serial(buf: &[i8], initial: i8) -> i8 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") { return unsafe { max_i8_avx2(buf, initial) }; }
        if is_x86_feature_detected!("avx") { return unsafe { max_i8_avx1(buf, initial) }; }
    }
    buf.iter().fold(initial, |a, &b| a.max(b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_i8_avx2(buf: &[i8], initial: i8) -> i8 {
    let mut max_v = _mm256_set1_epi8(initial); let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) { max_v = _mm256_max_epi8(max_v, _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i)); }
    let mut tmp = [0i8; 32]; _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n32..] { m = m.max(x); } m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_i8_avx1(buf: &[i8], initial: i8) -> i8 {
    let mut max_v = _mm_set1_epi8(initial); let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) { max_v = _mm_max_epi8(max_v, _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i)); }
    let mut tmp = [0i8; 16]; _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n16..] { m = m.max(x); } m
}
