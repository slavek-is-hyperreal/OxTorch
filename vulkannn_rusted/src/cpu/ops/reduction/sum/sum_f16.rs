#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sum reduction for F16 tensors.
pub fn sum_f16(buf: &[half::f16]) -> f32 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_f16_serial(chunk) as f64).sum::<f64>() as f32;
    }
    sum_f16_serial(buf)
}

fn sum_f16_serial(buf: &[half::f16]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { sum_f16_f16c(buf) };
        }
    }
    buf.iter().map(|x| x.to_f32() as f64).sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn sum_f16_f16c(buf: &[half::f16]) -> f32 {
    let mut sum_v = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        sum_v = _mm256_add_ps(sum_v, _mm256_cvtph_ps(h_vec));
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().map(|&x| x as f64).sum::<f64>(); for &x in &buf[n8..] { s += x.to_f32() as f64; } s as f32
}
