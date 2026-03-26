#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sum reduction for I8 tensors with multi-architecture support.
pub fn sum_i8(buf: &[i8]) -> i64 {
    const PAR_LIMIT: usize = 256_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(PAR_LIMIT).map(|c| sum_i8_serial(c)).sum();
    }
    sum_i8_serial(buf)
}

fn sum_i8_serial(buf: &[i8]) -> i64 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx512f") { return unsafe { sum_i8_avx512(buf) }; }
        if is_x86_feature_detected!("avx2") { return unsafe { sum_i8_avx2(buf) }; }
        if is_x86_feature_detected!("avx") { return unsafe { sum_i8_avx1(buf) }; }
    }
    // GPR fallback
    buf.iter().map(|&x| x as i64).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_i8_avx512(buf: &[i8]) -> i64 {
    let mut sum_v = _mm512_setzero_si512(); let n64 = (buf.len() / 64) * 64;
    for i in (0..n64).step_by(64) {
        let v = _mm512_loadu_si512(buf.as_ptr().add(i) as *const __m512i);
        let lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v));
        let hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(v, 1));
        sum_v = _mm512_add_epi32(sum_v, _mm512_add_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(lo)), _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(lo, 1))));
        sum_v = _mm512_add_epi32(sum_v, _mm512_add_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(hi)), _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(hi, 1))));
    }
    let s = _mm512_reduce_add_epi32(sum_v); 
    let mut s_acc = s as i64; for &b in &buf[n64..] { s_acc += b as i64; } s_acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_i8_avx2(buf: &[i8]) -> i64 {
    let mut sum_v = _mm256_setzero_si256(); let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let v = _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i);
        let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
        let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
        sum_v = _mm256_add_epi32(sum_v, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo, 1))));
        sum_v = _mm256_add_epi32(sum_v, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi, 1))));
    }
    let mut tmp = [0i32; 8]; _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum_v);
    let mut s = tmp.iter().map(|&x| x as i64).sum::<i64>(); for &b in &buf[n32..] { s += b as i64; } s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_i8_avx1(buf: &[i8]) -> i64 {
    let mut sum_v = _mm_setzero_si128(); let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) {
        let v = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        let lo = _mm_cvtepi8_epi16(v);
        let hi = _mm_cvtepi8_epi16(_mm_srli_si128(v, 8));
        sum_v = _mm_add_epi32(sum_v, _mm_add_epi32(_mm_cvtepi16_epi32(lo), _mm_cvtepi16_epi32(_mm_srli_si128(lo, 8))));
        sum_v = _mm_add_epi32(sum_v, _mm_add_epi32(_mm_cvtepi16_epi32(hi), _mm_cvtepi16_epi32(_mm_srli_si128(hi, 8))));
    }
    let mut tmp = [0i32; 4]; _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum_v);
    let mut s = tmp.iter().map(|&x| x as i64).sum::<i64>(); for &b in &buf[n16..] { s += b as i64; } s
}
