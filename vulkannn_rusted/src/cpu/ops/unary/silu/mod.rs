/// Elementwise SiLU activation for F32 tensors.
pub fn silu_f32(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        let v = *x;
        *x = v / (1.0 + (-v).exp());
    }
}

pub fn silu_f16(buf: &mut [half::f16]) {
    for x in buf.iter_mut() {
        let v = x.to_f32();
        *x = half::f16::from_f32(v / (1.0 + (-v).exp()));
    }
}

pub fn silu_bf16(buf: &mut [half::bf16]) {
    for x in buf.iter_mut() {
        let v = x.to_f32();
        *x = half::bf16::from_f32(v / (1.0 + (-v).exp()));
    }
}

pub fn silu_i8(buf: &mut [i8]) {
    // Scalar only for now
    for x in buf.iter_mut() {
        let v = *x as f32 / 16.0;
        let res = v / (1.0 + (-v).exp());
        *x = (res * 16.0).round() as i8;
    }
}
