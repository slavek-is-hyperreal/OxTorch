use rayon::prelude::*;

pub fn rms_norm_bf16(x: &[half::bf16], w: &[half::bf16], out: &mut [half::bf16], n: usize, d: usize, eps: f32) {
    // Convert weights once
    let mut w_f32 = crate::tensor::pool::TensorPool::get_f32_buffer(w.len());
    for i in 0..w.len() { w_f32[i] = w[i].to_f32(); }

    if n > 1 {
        x.par_chunks(d).zip(out.par_chunks_mut(d)).for_each(|(x_row, out_row)| {
            rms_norm_bf16_row(x_row, &w_f32, out_row, d, eps);
        });
    } else {
        rms_norm_bf16_row(x, &w_f32, out, d, eps);
    }
}

fn rms_norm_bf16_row(x: &[half::bf16], w_f32: &[f32], out: &mut [half::bf16], d: usize, eps: f32) {
    let mut x_f32 = crate::tensor::pool::TensorPool::get_f32_buffer(d);
    let mut out_f32 = crate::tensor::pool::TensorPool::get_f32_buffer(d);
    
    for i in 0..d { x_f32[i] = x[i].to_f32(); }
    
    crate::cpu::ops::norm::rms_norm::rms_norm_f32::rms_norm_f32(&x_f32, w_f32, &mut out_f32, 1, d, eps);
    
    for i in 0..d { out[i] = half::bf16::from_f32(out_f32[i]); }
}
