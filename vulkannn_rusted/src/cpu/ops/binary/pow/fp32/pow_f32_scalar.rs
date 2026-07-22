//! Baseline Scalar Implementation for FP32 Pow (tensor ^ scalar exponent).
//! Transcribed from cpu_old/ops/unary/pow/pow_f32.rs. Semantic reference for
//! both the general `powf` path and the exponent==2.0 fast path.

#[inline(always)]
pub fn pow(in_buf: &[f32], out_buf: &mut [f32], exponent: f32) {
    if exponent == 2.0 {
        for i in 0..in_buf.len() {
            out_buf[i] = in_buf[i] * in_buf[i];
        }
    } else {
        for i in 0..in_buf.len() {
            out_buf[i] = in_buf[i].powf(exponent);
        }
    }
}

#[inline(always)]
pub fn pow_inplace(buf: &mut [f32], exponent: f32) {
    if exponent == 2.0 {
        for x in buf.iter_mut() {
            *x = *x * *x;
        }
    } else {
        for x in buf.iter_mut() {
            *x = x.powf(exponent);
        }
    }
}
