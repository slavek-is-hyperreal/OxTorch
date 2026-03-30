/// Softmax reduction for I8 tensors (dequantized).
pub fn softmax_i8(buf: &mut [i8], is_log: bool) {
    if buf.is_empty() { return; }
    // Dequantize to f32 for accurate softmax using pooled workspace
    let mut f32_buf = crate::tensor::pool::TensorPool::get_buffer::<f32>(buf.len());
    for i in 0..buf.len() {
        f32_buf[i] = buf[i] as f32;
    }
    
    super::softmax_f32::softmax_f32(&mut f32_buf, is_log);
    
    // Re-quantize to i8 (truncate to match PyTorch .to(torch.int8))
    for i in 0..buf.len() {
        buf[i] = f32_buf[i].clamp(-128.0, 127.0) as i8;
    }
}
