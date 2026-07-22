//! Baseline Scalar ReLU for F16 (via f32). Transcribed from cpu_old relu_f16.

#[inline(always)]
pub fn relu(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    for i in 0..in_buf.len() {
        out_buf[i] = half::f16::from_f32(in_buf[i].to_f32().max(0.0));
    }
}

#[inline(always)]
pub fn relu_inplace(buf: &mut [half::f16]) {
    for x in buf.iter_mut() {
        *x = half::f16::from_f32(x.to_f32().max(0.0));
    }
}
