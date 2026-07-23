//! Unary elementwise ops (Wave 2). Category-level re-exports.

pub mod relu;
pub mod neg;
pub mod exp;
pub mod sigmoid;
pub mod silu;
pub mod tanh;
pub mod gelu;

pub use relu::{
    relu_f32, relu_f16, relu_bf16, relu_i8,
    relu_f32_inplace, relu_f16_inplace, relu_bf16_inplace, relu_i8_inplace,
    relu_f32_serial, relu_f16_serial, relu_bf16_serial, relu_i8_serial,
    relu_f32_inplace_serial, relu_f16_inplace_serial, relu_bf16_inplace_serial,
    relu_i8_inplace_serial,
};
pub use neg::{
    neg_f32, neg_f16, neg_bf16,
    neg_f32_inplace, neg_f16_inplace, neg_bf16_inplace,
    neg_f32_serial, neg_f16_serial, neg_bf16_serial,
    neg_f32_inplace_serial, neg_f16_inplace_serial, neg_bf16_inplace_serial,
};
pub use exp::{
    exp_f32, exp_f16, exp_bf16,
    exp_f32_inplace, exp_f16_inplace, exp_bf16_inplace,
    exp_f32_serial, exp_f32_inplace_serial,
    exp_f16_inplace_serial, exp_bf16_inplace_serial,
};
pub use sigmoid::{
    sigmoid_f32, sigmoid_f16, sigmoid_bf16, sigmoid_i8,
    sigmoid_f32_inplace, sigmoid_f16_inplace, sigmoid_bf16_inplace,
    sigmoid_f32_serial, sigmoid_f32_inplace_serial,
    sigmoid_f16_inplace_serial, sigmoid_bf16_inplace_serial,
};
pub use silu::{
    silu_f32, silu_f16, silu_bf16, silu_i8,
    silu_f32_inplace, silu_f16_inplace, silu_bf16_inplace,
    silu_f32_serial, silu_f32_inplace_serial,
    silu_f16_inplace_serial, silu_bf16_inplace_serial,
};
pub use tanh::{
    tanh_f32, tanh_f16, tanh_bf16, tanh_i8,
    tanh_f32_inplace, tanh_f16_inplace, tanh_bf16_inplace,
    tanh_f32_serial, tanh_f32_inplace_serial,
    tanh_f16_inplace_serial, tanh_bf16_inplace_serial,
};
pub use gelu::{
    gelu_f32, gelu_f16, gelu_bf16, gelu_i8,
    gelu_f32_inplace, gelu_f16_inplace, gelu_bf16_inplace,
    gelu_f32_serial, gelu_f32_inplace_serial,
    gelu_f16_inplace_serial, gelu_bf16_inplace_serial,
};
