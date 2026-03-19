#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn relu_i8_inplace(buf: &mut [i8]) {
    relu_i8_swar(buf);
}

pub fn relu_i8_swar(buf: &mut [i8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { relu_i8_avx2(buf) };
    }
    let mut chunks = buf.chunks_exact_mut(8);
    for chunk in chunks.by_ref() {
        let word: u64 = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const u64) };
        let is_neg = word & 0x8080808080808080;
        let mask = (is_neg >> 7).wrapping_mul(0xFF);
        unsafe { std::ptr::write_unaligned(chunk.as_mut_ptr() as *mut u64, word & !mask); }
    }
    for b in chunks.into_remainder() {
        if *b < 0 { *b = 0; }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn relu_i8_avx2(buf: &mut [i8]) {
    let zero = _mm256_setzero_si256();
    let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let v = _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(buf.as_mut_ptr().add(i) as *mut __m256i, _mm256_max_epi8(v, zero));
    }
    for b in &mut buf[n32..] {
        if *b < 0 { *b = 0; }
    }
}
