//! Scalar LayerNorm per-row. f32 accumulation (Rule 6: norm stays f32, unlike
//! the sum reduction which is f64). std `sqrt` (hardware, ~0.5 ULP — no rsqrt
//! approximation). Transcribed from cpu_old/ops/norm/layer_norm/layer_norm_f32.rs.
pub fn row(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], eps: f32) {
    let d = x.len();
    let mut sum = 0.0f32;
    for &v in x { sum += v; }
    let mean = sum / d as f32;
    let mut var_sum = 0.0f32;
    for &v in x { let df = v - mean; var_sum += df * df; }
    let inv_std = 1.0 / (var_sum / d as f32 + eps).sqrt();
    let (hw, hb) = (!w.is_empty(), !b.is_empty());
    for i in 0..d {
        let weight = if hw { w[i] } else { 1.0 };
        let bias = if hb { b[i] } else { 0.0 };
        out[i] = (x[i] - mean) * inv_std * weight + bias;
    }
}
