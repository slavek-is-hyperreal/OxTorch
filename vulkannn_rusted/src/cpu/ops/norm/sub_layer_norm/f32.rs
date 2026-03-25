use crate::tensor::error::PyResult;

/// Sub-LayerNorm for F32 fallback. SubLN zeroes out the mean before normalizing.
pub fn sub_layer_norm_f32(
    x: &[f32],
    weight: Option<&[f32]>,
    eps: f32,
    normalized_shape: &[usize],  // Usually [hidden_dim]
    out: &mut [f32]
) -> PyResult<()> {
    // Basic CPU implementation.
    let hidden_dim = normalized_shape[normalized_shape.len() - 1];
    let num_tokens = x.len() / hidden_dim;

    for i in 0..num_tokens {
        let x_slice = &x[i * hidden_dim..(i + 1) * hidden_dim];
        let out_slice = &mut out[i * hidden_dim..(i + 1) * hidden_dim];

        // 1. Calculate Mean
        let mut sum = 0.0;
        for &val in x_slice {
            sum += val;
        }
        let mean = sum / hidden_dim as f32;

        // 2. Subtract Mean and calculate Variance
        let mut var_sum = 0.0;
        for j in 0..hidden_dim {
            let zero_centered = x_slice[j] - mean;
            out_slice[j] = zero_centered; // store temporarily
            var_sum += zero_centered * zero_centered;
        }
        let variance = var_sum / hidden_dim as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // 3. Normalize and Apply Weight
        if let Some(w) = weight {
            for j in 0..hidden_dim {
                out_slice[j] = out_slice[j] * inv_std * w[j];
            }
        } else {
            for j in 0..hidden_dim {
                out_slice[j] = out_slice[j] * inv_std;
            }
        }
    }
    Ok(())
}
