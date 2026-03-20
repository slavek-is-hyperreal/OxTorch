/// Elementwise GELU activation for I8 tensors (LUT-based).
pub fn gelu_i8(buf: &mut [i8]) {
    static mut GELU_LUT: [i8; 256] = [0; 256];
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        for i in 0..256 {
            let x = (i as i32 - 128) as f32;
            let res = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
            // Use truncation (as i8) to match PyTorch's .to(torch.int8) cast behavior in benchmarks
            unsafe { GELU_LUT[i] = res.clamp(-128.0, 127.0) as i8; }
        }
    });
    for x in buf.iter_mut() {
        *x = unsafe { GELU_LUT[(*x as i32 + 128) as usize] };
    }
}
