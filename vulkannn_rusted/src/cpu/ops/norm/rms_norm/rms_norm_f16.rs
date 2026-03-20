use rayon::prelude::*;

pub fn rms_norm_f16(x: &[half::f16], w: &[half::f16], out: &mut [half::f16], n: usize, d: usize, eps: f32) {
    if n > 1 {
        x.par_chunks(d).zip(out.par_chunks_mut(d)).for_each(|(x_row, out_row)| {
            rms_norm_f16_row(x_row, w, out_row, d, eps);
        });
    } else {
        rms_norm_f16_row(x, w, out, d, eps);
    }
}

fn rms_norm_f16_row(x: &[half::f16], w: &[half::f16], out: &mut [half::f16], d: usize, eps: f32) {
    let mut x_f32 = vec![0.0; d];
    let mut w_f32 = vec![0.0; w.len()];
    for i in 0..d { x_f32[i] = x[i].to_f32(); }
    for i in 0..w.len() { w_f32[i] = w[i].to_f32(); }
    
    let mut out_f32 = vec![0.0; d];
    crate::cpu::ops::norm::rms_norm::rms_norm_f32::rms_norm_f32(&x_f32, &w_f32, &mut out_f32, 1, d, eps);
    
    for i in 0..d { out[i] = half::f16::from_f32(out_f32[i]); }
}
