pub mod exp_f32;
pub use exp_f32::*;

pub fn exp_f16(buf: &mut [half::f16]) {
    let mut f32_buf = crate::tensor::pool::TensorPool::get_f32_buffer(buf.len());
    for i in 0..buf.len() { f32_buf[i] = buf[i].to_f32(); }
    exp_f32(&mut f32_buf);
    for i in 0..buf.len() { buf[i] = half::f16::from_f32(f32_buf[i]); }
}

pub fn exp_bf16(buf: &mut [half::bf16]) {
    let mut f32_buf = crate::tensor::pool::TensorPool::get_f32_buffer(buf.len());
    for i in 0..buf.len() { f32_buf[i] = buf[i].to_f32(); }
    exp_f32(&mut f32_buf);
    for i in 0..buf.len() { buf[i] = half::bf16::from_f32(f32_buf[i]); }
}
