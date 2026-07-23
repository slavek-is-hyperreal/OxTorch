//! Baseline Scalar gelu (tanh-approx) for FP32.
//! `0.5·x·(1 + tanh(K·(x + C·x³)))`, K=√(2/π), C=0.044715.

const K: f32 = 0.7978845608;
const C: f32 = 0.044715;

#[inline(always)]
pub fn gelu_one(x: f32) -> f32 {
    let inner = K * (x + C * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline(always)]
pub fn gelu(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = gelu_one(in_buf[i]);
    }
}

#[inline(always)]
pub fn gelu_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = gelu_one(*x);
    }
}
