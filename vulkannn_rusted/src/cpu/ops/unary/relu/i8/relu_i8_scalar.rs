//! Baseline Scalar ReLU for I8 (max(x, 0)). Transcribed from cpu_old relu_i8.

#[inline(always)]
pub fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
    for i in 0..in_buf.len() {
        out_buf[i] = in_buf[i].max(0i8);
    }
}

#[inline(always)]
pub fn relu_inplace(buf: &mut [i8]) {
    for x in buf.iter_mut() {
        *x = (*x).max(0i8);
    }
}
