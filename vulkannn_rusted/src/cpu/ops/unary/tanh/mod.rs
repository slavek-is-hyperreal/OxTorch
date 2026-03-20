/// Elementwise Tanh activation for F32 tensors.
pub fn tanh_f32(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = x.tanh();
    }
}

pub fn tanh_f16(buf: &mut [half::f16]) {
    for x in buf.iter_mut() {
        *x = half::f16::from_f32(x.to_f32().tanh());
    }
}

pub fn tanh_bf16(buf: &mut [half::bf16]) {
    for x in buf.iter_mut() {
        *x = half::bf16::from_f32(x.to_f32().tanh());
    }
}

pub fn tanh_i8(buf: &mut [i8]) {
    for x in buf.iter_mut() {
        let v = *x as f32 / 16.0;
        let res = v.tanh();
        *x = (res * 16.0).round() as i8;
    }
}
