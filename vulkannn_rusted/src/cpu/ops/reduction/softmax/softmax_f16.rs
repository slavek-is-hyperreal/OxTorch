/// Softmax reduction for F16 tensors.
pub fn softmax_f16(buf: &mut [half::f16], is_log: bool) {
    if buf.is_empty() { return; }
    let max_val = buf.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.to_f32()));
    let mut sum = 0.0f32;
    for x in buf.iter() { sum += (x.to_f32() - max_val).exp(); }
    
    if is_log {
        let log_sum = sum.ln();
        for x in buf.iter_mut() { *x = half::f16::from_f32(x.to_f32() - max_val - log_sum); }
    } else {
        let inv_sum = 1.0 / sum;
        for x in buf.iter_mut() { *x = half::f16::from_f32((x.to_f32() - max_val).exp() * inv_sum); }
    }
}
