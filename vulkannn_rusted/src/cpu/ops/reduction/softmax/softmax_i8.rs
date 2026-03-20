/// Softmax reduction for I8 tensors (dequantized).
pub fn softmax_i8(buf: &mut [i8], is_log: bool) {
    if buf.is_empty() { return; }
    // Dequantize to f32 for accurate softmax
    let mut f32_buf: Vec<f32> = buf.iter().map(|&x| x as f32).collect();
    crate::cpu::softmax_f32(&mut f32_buf, is_log);
    // Re-quantize to i8 (truncate to match PyTorch .to(torch.int8))
    for i in 0..buf.len() {
        buf[i] = f32_buf[i].clamp(-128.0, 127.0) as i8;
    }
}
