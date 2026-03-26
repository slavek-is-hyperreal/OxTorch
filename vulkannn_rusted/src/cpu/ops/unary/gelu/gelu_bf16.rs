/// Elementwise GELU activation for BF16 tensors.
pub fn gelu_bf16(buf: &mut [half::bf16]) {
    const CHUNK_SIZE: usize = 1024; 
    let mut tmp = [0.0f32; CHUNK_SIZE];
    for chunk in buf.chunks_mut(CHUNK_SIZE) {
        for (i, x) in chunk.iter().enumerate() { tmp[i] = x.to_f32(); }
        super::gelu_f32::gelu_f32(&mut tmp[..chunk.len()]);
        for (i, x) in chunk.iter_mut().enumerate() { *x = half::bf16::from_f32(tmp[i]); }
    }
}
