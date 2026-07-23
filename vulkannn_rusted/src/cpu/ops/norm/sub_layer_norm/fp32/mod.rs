//! Scalar SubLN (Sub-LayerNorm): subtract mean, normalize by std, apply weight
//! (NO bias). f32 accumulation, std sqrt. Transcribed VERBATIM from cpu_old
//! sub_layer_norm/f32.rs.
use pyo3::prelude::PyResult;
pub fn sub_layer_norm_f32(x: &[f32], weight: Option<&[f32]>, eps: f32, normalized_shape: &[usize], out: &mut [f32]) -> PyResult<()> {
    let hidden = normalized_shape[normalized_shape.len() - 1];
    let ntok = x.len() / hidden;
    for i in 0..ntok {
        let xs = &x[i * hidden..(i + 1) * hidden];
        let os = &mut out[i * hidden..(i + 1) * hidden];
        let mut sum = 0.0f32; for &v in xs { sum += v; }
        let mean = sum / hidden as f32;
        let mut vs = 0.0f32;
        for j in 0..hidden { let zc = xs[j] - mean; os[j] = zc; vs += zc * zc; }
        let inv_std = 1.0 / (vs / hidden as f32 + eps).sqrt();
        if let Some(w) = weight { for j in 0..hidden { os[j] = os[j] * inv_std * w[j]; } }
        else { for j in 0..hidden { os[j] *= inv_std; } }
    }
    Ok(())
}
