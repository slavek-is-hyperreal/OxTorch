//! AVX1 LayerNorm per-row. Transcribed from cpu_old layer_norm_f32_row_avx.
//! f32 accumulation; std sqrt. BENCH: memory-bound (§8); not re-benchmarked.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn row(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], eps: f32) {
    let d = x.len(); let d8 = (d / 8) * 8;
    let mut s = _mm256_setzero_ps();
    for i in (0..d8).step_by(8) { s = _mm256_add_ps(s, _mm256_loadu_ps(x.as_ptr().add(i))); }
    let mut arr = [0.0f32; 8]; _mm256_storeu_ps(arr.as_mut_ptr(), s);
    let mut sum = arr.iter().sum::<f32>();
    for i in d8..d { sum += x[i]; }
    let mean = sum / d as f32;
    let meanv = _mm256_set1_ps(mean);
    let mut vv = _mm256_setzero_ps();
    for i in (0..d8).step_by(8) {
        let df = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), meanv);
        vv = _mm256_add_ps(vv, _mm256_mul_ps(df, df));
    }
    _mm256_storeu_ps(arr.as_mut_ptr(), vv);
    let mut var = arr.iter().sum::<f32>();
    for i in d8..d { let df = x[i] - mean; var += df * df; }
    let inv_std = 1.0 / (var / d as f32 + eps).sqrt();
    let isv = _mm256_set1_ps(inv_std);
    let (hw, hb) = (!w.is_empty(), !b.is_empty());
    for i in (0..d8).step_by(8) {
        let val = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), meanv), isv);
        let weight = if hw { _mm256_loadu_ps(w.as_ptr().add(i)) } else { _mm256_set1_ps(1.0) };
        let bias = if hb { _mm256_loadu_ps(b.as_ptr().add(i)) } else { _mm256_setzero_ps() };
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(_mm256_mul_ps(val, weight), bias));
    }
    for i in d8..d {
        let weight = if hw { w[i] } else { 1.0 };
        let bias = if hb { b[i] } else { 0.0 };
        out[i] = (x[i] - mean) * inv_std * weight + bias;
    }
}
