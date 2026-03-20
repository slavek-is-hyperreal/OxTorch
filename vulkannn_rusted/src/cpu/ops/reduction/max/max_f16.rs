#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Max reduction for F16 tensors.
pub fn max_f16(buf: &[half::f16], initial: f32) -> f32 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(128_000).map(|c| max_f16_serial(c, initial)).reduce(|| initial, |a, b| a.max(b));
    }
    max_f16_serial(buf, initial)
}

fn max_f16_serial(buf: &[half::f16], initial: f32) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { max_f16_f16c(buf, initial) };
        }
    }
    buf.iter().fold(initial, |a, &b| a.max(b.to_f32()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn max_f16_f16c(buf: &[half::f16], initial: f32) -> f32 {
    let mut max_v = _mm256_set1_ps(initial); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        max_v = _mm256_max_ps(max_v, _mm256_cvtph_ps(h_vec));
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n8..] { m = m.max(x.to_f32()); } m
}
