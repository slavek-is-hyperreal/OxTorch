#[cfg(target_arch = "x86_64")]


/// Elementwise Sigmoid activation for F32 tensors.
pub fn sigmoid_f32(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

pub fn sigmoid_f16(buf: &mut [half::f16]) {
    for x in buf.iter_mut() {
        let v = x.to_f32();
        *x = half::f16::from_f32(1.0 / (1.0 + (-v).exp()));
    }
}

pub fn sigmoid_bf16(buf: &mut [half::bf16]) {
    for x in buf.iter_mut() {
        let v = x.to_f32();
        *x = half::bf16::from_f32(1.0 / (1.0 + (-v).exp()));
    }
}

pub fn sigmoid_i8(buf: &mut [i8]) {
    static mut SIGMOID_LUT: [i8; 256] = [0; 256];
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        for i in 0..256 {
            let x = (i as i32 - 128) as f32;
            let res = 127.0 / (1.0 + (-x / 16.0).exp()); // Simplified i8-range sigmoid
            unsafe { SIGMOID_LUT[i] = res.round() as i8; }
        }
    });
    for x in buf.iter_mut() {
        *x = unsafe { SIGMOID_LUT[(*x as i32 + 128) as usize] };
    }
}
