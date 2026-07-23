//! Scalar RMSNorm per-row. f32 accumulation, std sqrt. out = x*inv_rms*w.
//! Transcribed from cpu_old rms_norm_f32_row_scalar.
pub fn row(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let d = x.len();
    let mut sq = 0.0f32;
    for &v in x { sq += v * v; }
    let inv_rms = 1.0 / (sq / d as f32 + eps).sqrt();
    let hw = !w.is_empty();
    for i in 0..d {
        let weight = if hw { w[i] } else { 1.0 };
        out[i] = x[i] * inv_rms * weight;
    }
}
