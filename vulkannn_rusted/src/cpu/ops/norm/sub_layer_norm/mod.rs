//! SUB_LAYER_NORM — f32 only (legacy set). SubLN zeroes the mean before norm.
pub mod fp32;
pub use fp32::sub_layer_norm_f32;
