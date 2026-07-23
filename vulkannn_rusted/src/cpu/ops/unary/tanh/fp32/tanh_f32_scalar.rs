//! Baseline Scalar tanh for FP32 — std `f32::tanh()` (~1 ULP), the reference.

#[inline(always)]
pub fn tanh(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = in_buf[i].tanh();
    }
}

#[inline(always)]
pub fn tanh_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = x.tanh();
    }
}
