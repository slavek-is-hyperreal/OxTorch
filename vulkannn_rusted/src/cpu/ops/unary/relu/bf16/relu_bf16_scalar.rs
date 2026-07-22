//! Baseline Scalar ReLU for BF16 (via f32). Transcribed from cpu_old relu_bf16.

#[inline(always)]
pub fn relu(in_buf: &[half::bf16], out_buf: &mut [half::bf16]) {
    for i in 0..in_buf.len() {
        out_buf[i] = half::bf16::from_f32(in_buf[i].to_f32().max(0.0));
    }
}

#[inline(always)]
pub fn relu_inplace(buf: &mut [half::bf16]) {
    for x in buf.iter_mut() {
        *x = half::bf16::from_f32(x.to_f32().max(0.0));
    }
}
