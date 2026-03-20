#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sum reduction for F32 tensors with AVX-512 and AVX support.
pub fn sum_f32(buf: &[f32]) -> f32 {
    const PAR_LIMIT: usize = 256_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_f32_serial(chunk)).sum();
    }
    sum_f32_serial(buf)
}

fn sum_f32_serial(buf: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        unsafe {
            if is_x86_feature_detected!("avx512f") { return sum_f32_avx512(buf); }
            if is_x86_feature_detected!("avx") { return sum_f32_avx(buf); }
        }
    }
    buf.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f32_avx512(buf: &[f32]) -> f32 {
    let mut sum_v = _mm512_setzero_ps(); let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) { sum_v = _mm512_add_ps(sum_v, _mm512_loadu_ps(buf.as_ptr().add(i))); }
    let mut tmp = [0.0f32; 16]; _mm512_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().sum::<f32>(); for &x in &buf[n16..] { s += x; } s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_f32_avx(buf: &[f32]) -> f32 {
    let mut sum_v = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) { sum_v = _mm256_add_ps(sum_v, _mm256_loadu_ps(buf.as_ptr().add(i))); }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().sum::<f32>(); for &x in &buf[n8..] { s += x; } s
}
