//! Baseline Scalar sigmoid for FP32. Numerically-stable form (exp is always
//! evaluated on -|x| <= 0, so it stays in [0,1] and never overflows; the naive
//! 1/(1+exp(-x)) flushes the denormal tail to 0 for large negative x):
//!   z = exp(-|x|);  sigmoid = (x < 0 ? z : 1) / (1 + z)

#[inline(always)]
pub fn sigmoid_one(x: f32) -> f32 {
    let z = (-x.abs()).exp();
    let num = if x < 0.0 { z } else { 1.0 };
    num / (1.0 + z)
}

#[inline(always)]
pub fn sigmoid(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = sigmoid_one(in_buf[i]);
    }
}

#[inline(always)]
pub fn sigmoid_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = sigmoid_one(*x);
    }
}
