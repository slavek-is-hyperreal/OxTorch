//! Baseline Scalar ReLU for FP32 (max(x, 0)). Semantic reference.
//! Transcribed from cpu_old/ops/unary/relu/relu_f32.rs.

#[inline(always)]
pub fn relu(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = in_buf[i].max(0.0);
    }
}

#[inline(always)]
pub fn relu_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = x.max(0.0);
    }
}
