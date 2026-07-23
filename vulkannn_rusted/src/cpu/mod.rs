pub mod ops;
pub mod swar;
pub mod conversions;
pub mod tiling_cpu;
pub mod dispatch;
pub mod thresholds;
pub mod simd_util;
#[cfg(test)]
pub mod parity_harness;

// Bridge to cpu_old for incremental migration. 
// Symbols in this module (src/cpu) will shadow those in cpu_old.
pub use crate::cpu_old::*;

// New granular CPU backend re-exports
pub use ops::binary::add::{add_bf16, add_f16, add_i8};
pub use ops::binary::sub::{sub_bf16, sub_f16, sub_i8};
pub use ops::binary::mul::{mul_bf16, mul_f32, mul_f16, mul_i8, mul_broadcast_f32};
pub use ops::binary::div::{div_bf16, div_f32, div_f16, div_i8};
pub use ops::binary::atan2::atan2_f32;
pub use ops::binary::pow::{pow_f32, pow_f32_inplace};
pub use ops::binary::scalar::{scalar_op_f32, scalar_op_f16, scalar_op_bf16, scalar_op_i8};
pub use ops::unary::relu::{
    relu_f32, relu_f16, relu_bf16, relu_i8,
    relu_f32_inplace, relu_f16_inplace, relu_bf16_inplace, relu_i8_inplace,
};
pub use ops::unary::neg::{
    neg_f32, neg_f16, neg_bf16,
    neg_f32_inplace, neg_f16_inplace, neg_bf16_inplace,
};
pub use ops::unary::exp::{
    exp_f32, exp_f16, exp_bf16,
    exp_f32_inplace, exp_f16_inplace, exp_bf16_inplace,
};
pub use ops::unary::sigmoid::{
    sigmoid_f32, sigmoid_f16, sigmoid_bf16, sigmoid_i8,
    sigmoid_f32_inplace, sigmoid_f16_inplace, sigmoid_bf16_inplace,
};
pub use ops::unary::silu::{
    silu_f32, silu_f16, silu_bf16, silu_i8,
    silu_f32_inplace, silu_f16_inplace, silu_bf16_inplace,
};
