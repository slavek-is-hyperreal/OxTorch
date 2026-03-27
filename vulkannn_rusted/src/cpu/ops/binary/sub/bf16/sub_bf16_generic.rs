/// Elementwise subtraction for BF16 (Scalar fallback).
pub fn sub_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    for i in 0..a.len() {
        res[i] = half::bf16::from_f32(a[i].to_f32() - b[i].to_f32());
    }
}
