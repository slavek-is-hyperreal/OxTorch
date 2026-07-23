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
pub use ops::unary::tanh::{
    tanh_f32, tanh_f16, tanh_bf16, tanh_i8,
    tanh_f32_inplace, tanh_f16_inplace, tanh_bf16_inplace,
};
pub use ops::unary::gelu::{
    gelu_f32, gelu_f16, gelu_bf16, gelu_i8,
    gelu_f32_inplace, gelu_f16_inplace, gelu_bf16_inplace,
};
pub use ops::reduction::sum::{sum_f32, sum_f16, sum_bf16, sum_i8, sum_f32_acc, sum_f16_acc, sum_bf16_acc, sum_i8_acc};
pub use ops::reduction::max::{max_f32, max_f16, max_bf16, max_i8};
pub use ops::reduction::argmax::{argmax_f32, argmax_f16};
pub use ops::reduction::softmax::{softmax_f32, softmax_f16, softmax_bf16, softmax_i8};
pub use ops::norm::layer_norm::{layer_norm_f32, layer_norm_f16, layer_norm_bf16};
pub use ops::norm::rms_norm::{rms_norm_f32, rms_norm_f16, rms_norm_bf16};
pub use ops::norm::sub_layer_norm::sub_layer_norm_f32;
pub use ops::sequence::cat::{cat_f32, cat_f16, cat_bf16, cat_i8};
pub use ops::indexing::index_select::{index_select_f32, index_select_f16, index_select_bf16, index_select_i8, embedding_f32};
