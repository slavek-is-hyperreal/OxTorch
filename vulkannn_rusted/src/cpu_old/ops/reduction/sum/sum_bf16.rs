#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sum reduction for BF16 tensors.
pub fn sum_bf16(buf: &[half::bf16]) -> f32 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_bf16_serial(chunk) as f64).sum::<f64>() as f32;
    }
    sum_bf16_serial(buf)
}

fn sum_bf16_serial(buf: &[half::bf16]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { sum_bf16_avx(buf) }; }
    }
    buf.iter().map(|x| x.to_f32() as f64).sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_bf16_avx(buf: &[half::bf16]) -> f32 {
    let mut sum_v = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    let zero128 = _mm_setzero_si128();
    for i in (0..n8).step_by(8) {
        let b_raw = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, b_raw)),
            _mm_unpackhi_epi16(zero128, b_raw)
        ));
        sum_v = _mm256_add_ps(sum_v, f_vec);
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().map(|&x| x as f64).sum::<f64>(); for &x in &buf[n8..] { s += x.to_f32() as f64; } s as f32
}
