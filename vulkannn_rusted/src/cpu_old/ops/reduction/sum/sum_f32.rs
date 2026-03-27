#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sum reduction for F32 tensors with AVX-512 and AVX support.
pub fn sum_f32(buf: &[f32]) -> f32 {
    const PAR_LIMIT: usize = 256_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_f32_serial(chunk) as f64).sum::<f64>() as f32;
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
    #[cfg(target_arch = "aarch64")] {
        return unsafe { sum_f32_neon(buf) };
    }
    buf.iter().map(|&x| x as f64).sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f32_avx512(buf: &[f32]) -> f32 {
    let mut s1 = _mm512_setzero_ps();
    let mut s2 = _mm512_setzero_ps();
    let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        s1 = _mm512_add_ps(s1, _mm512_loadu_ps(buf.as_ptr().add(i)));
        s2 = _mm512_add_ps(s2, _mm512_loadu_ps(buf.as_ptr().add(i + 16)));
    }
    let sum_v = _mm512_add_ps(s1, s2);
    let mut tmp = [0.0f32; 16]; _mm512_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().map(|&x| x as f64).sum::<f64>(); 
    for &x in &buf[n32..] { s += x as f64; } s as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_f32_avx(buf: &[f32]) -> f32 {
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut s4 = _mm256_setzero_ps();
    let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(buf.as_ptr().add(i)));
        s2 = _mm256_add_ps(s2, _mm256_loadu_ps(buf.as_ptr().add(i + 8)));
        s3 = _mm256_add_ps(s3, _mm256_loadu_ps(buf.as_ptr().add(i + 16)));
        s4 = _mm256_add_ps(s4, _mm256_loadu_ps(buf.as_ptr().add(i + 24)));
    }
    let sum_v = _mm256_add_ps(_mm256_add_ps(s1, s2), _mm256_add_ps(s3, s4));
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().map(|&x| x as f64).sum::<f64>(); 
    for &x in &buf[n32..] { s += x as f64; } s as f32
}

#[cfg(target_arch = "aarch64")]
unsafe fn sum_f32_neon(buf: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        s1 = vaddq_f32(s1, vld1q_f32(buf.as_ptr().add(i)));
        s2 = vaddq_f32(s2, vld1q_f32(buf.as_ptr().add(i + 4)));
    }
    let sum_v = vaddq_f32(s1, s2);
    let res = vaddvq_f32(sum_v);
    let mut s = res as f64;
    for &x in &buf[n8..] { s += x as f64; }
    s as f32
}
