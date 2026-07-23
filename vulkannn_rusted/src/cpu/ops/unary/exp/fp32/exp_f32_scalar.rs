//! Baseline Scalar exp for FP32 — uses std `f32::exp()` (~0.5 ULP, correctly
//! rounded), the semantic reference. See docs/kernel_specs/exp_spec.md.

#[inline(always)]
pub fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = in_buf[i].exp();
    }
}

#[inline(always)]
pub fn exp_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = x.exp();
    }
}
