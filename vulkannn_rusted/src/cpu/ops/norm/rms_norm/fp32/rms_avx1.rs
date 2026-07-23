//! AVX1 RMSNorm per-row. Transcribed from cpu_old rms_norm_f32_row_avx.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn row(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let d = x.len(); let d8 = (d / 8) * 8;
    let mut sq = _mm256_setzero_ps();
    for i in (0..d8).step_by(8) { let v = _mm256_loadu_ps(x.as_ptr().add(i)); sq = _mm256_add_ps(sq, _mm256_mul_ps(v, v)); }
    let mut arr = [0.0f32; 8]; _mm256_storeu_ps(arr.as_mut_ptr(), sq);
    let mut s = arr.iter().sum::<f32>();
    for i in d8..d { s += x[i] * x[i]; }
    let inv_rms = 1.0 / (s / d as f32 + eps).sqrt();
    let iv = _mm256_set1_ps(inv_rms);
    let hw = !w.is_empty();
    for i in (0..d8).step_by(8) {
        let val = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), iv);
        let weight = if hw { _mm256_loadu_ps(w.as_ptr().add(i)) } else { _mm256_set1_ps(1.0) };
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(val, weight));
    }
    for i in d8..d { let weight = if hw { w[i] } else { 1.0 }; out[i] = x[i] * inv_rms * weight; }
}
