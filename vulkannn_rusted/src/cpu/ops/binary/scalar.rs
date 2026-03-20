
/// Scalar operations for F32 tensors.
pub fn scalar_op_f32(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x + scalar; },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x - scalar; },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x * scalar; },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0.0 { out_buf[i] = x / scalar; } else { out_buf[i] = 0.0; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for F16 tensors.
pub fn scalar_op_f16(in_buf: &[half::f16], scalar: f32, out_buf: &mut [half::f16], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32() + scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32() - scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32() * scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0.0 { out_buf[i] = half::f16::from_f32(x.to_f32() / scalar); } else { out_buf[i] = half::f16::ZERO; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for BF16 tensors.
pub fn scalar_op_bf16(in_buf: &[half::bf16], scalar: f32, out_buf: &mut [half::bf16], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() + scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() - scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() * scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0.0 { out_buf[i] = half::bf16::from_f32(x.to_f32() / scalar); } else { out_buf[i] = half::bf16::ZERO; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for I8 tensors.
pub fn scalar_op_i8(in_buf: &[i8], scalar: i8, out_buf: &mut [i8], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_add(scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_sub(scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_mul(scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0 { out_buf[i] = x / scalar; } else { out_buf[i] = 0; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}
