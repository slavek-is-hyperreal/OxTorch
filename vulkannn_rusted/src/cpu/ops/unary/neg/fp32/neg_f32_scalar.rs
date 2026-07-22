//! Baseline Scalar negation for FP32. Transcribed from cpu_old neg_f32.

#[inline(always)]
pub fn neg(in_buf: &[f32], out_buf: &mut [f32]) {
    for i in 0..in_buf.len() {
        out_buf[i] = -in_buf[i];
    }
}

#[inline(always)]
pub fn neg_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = -*x;
    }
}
