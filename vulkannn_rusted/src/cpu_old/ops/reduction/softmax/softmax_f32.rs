/// Softmax reduction for F32 tensors.
pub fn softmax_f32(buf: &mut [f32], is_log: bool) {
    if buf.is_empty() { return; }
    let max_val = buf.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for x in buf.iter() { sum += (*x - max_val).exp(); }
    
    if is_log {
        let log_sum = sum.ln();
        for x in buf.iter_mut() { *x = *x - max_val - log_sum; }
    } else {
        let inv_sum = 1.0 / sum;
        for x in buf.iter_mut() { *x = (*x - max_val).exp() * inv_sum; }
    }
}
