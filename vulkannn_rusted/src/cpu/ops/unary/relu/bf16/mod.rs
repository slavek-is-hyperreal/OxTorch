//! BF16 ReLU — Tier II serial dispatcher. Legacy has an AVX1 out-of-place kernel
//! and a scalar in-place path only; transcribed faithfully (no bf16 SIMD in-place
//! kernel is fabricated — Rule 1).

pub mod relu_bf16_scalar;

#[cfg(target_arch = "x86_64")]
pub mod relu_bf16_avx;

pub fn relu(in_buf: &[half::bf16], out_buf: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { relu_bf16_avx::relu(in_buf, out_buf) };
        }
    }
    relu_bf16_scalar::relu(in_buf, out_buf);
}

pub fn relu_inplace(buf: &mut [half::bf16]) {
    relu_bf16_scalar::relu_inplace(buf);
}
