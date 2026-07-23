//! Baseline Scalar silu for FP32: x/(1+exp(-x)). Legacy's form (matches torch,
//! incl. the deep-tail flush to ∓0). See docs/kernel_specs/silu_spec.md.

#[inline(always)]
pub fn silu_one(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn silu(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = silu_one(in_buf[i]);
    }
}

#[inline(always)]
pub fn silu_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = silu_one(*x);
    }
}
