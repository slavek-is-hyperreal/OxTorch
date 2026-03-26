use rayon::prelude::*;

/// Argmax reduction for F32 tensors.
pub fn argmax_f32(in_buf: &[f32], out_buf: &mut [f32], _outer: usize, dim_size: usize, inner: usize) {
    out_buf.par_chunks_mut(inner).enumerate().for_each(|(i, out_row)| {
        for k in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            
            for j in 0..dim_size {
                let val = in_buf[i * dim_size * inner + j * inner + k];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            out_row[k] = max_idx as f32;
        }
    });
}

/// Argmax reduction for F16 tensors.
pub fn argmax_f16(in_buf: &[half::f16], out_buf: &mut [f32], _outer: usize, dim_size: usize, inner: usize) {
    out_buf.par_chunks_mut(inner).enumerate().for_each(|(i, out_row)| {
        for k in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            
            for j in 0..dim_size {
                let val = in_buf[i * dim_size * inner + j * inner + k].to_f32();
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            out_row[k] = max_idx as f32;
        }
    });
}
