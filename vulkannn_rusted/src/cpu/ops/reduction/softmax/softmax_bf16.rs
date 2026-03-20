/// Softmax reduction for BF16 tensors.
pub fn softmax_bf16(buf: &mut [half::bf16], is_log: bool) {
    if buf.is_empty() { return; }
    let max_val = super::super::max::max_bf16(buf, f32::NEG_INFINITY);
    use rayon::prelude::*;
    let sum: f32 = buf.par_chunks(64_000).map(|chunk| {
        let mut s = 0.0f32;
        for x in chunk { s += (x.to_f32() - max_val).exp(); }
        s
    }).sum();
    
    if is_log {
        let log_sum = sum.ln();
        buf.par_chunks_mut(64_000).for_each(|chunk| {
            for x in chunk { *x = half::bf16::from_f32(x.to_f32() - max_val - log_sum); }
        });
    } else {
        let inv_sum = 1.0 / sum;
        buf.par_chunks_mut(64_000).for_each(|chunk| {
            for x in chunk { *x = half::bf16::from_f32((x.to_f32() - max_val).exp() * inv_sum); }
        });
    }
}
