//! ReLU — Tier III parallel gate. Rayon lives ONLY here. Legacy relu had no
//! rayon; thresholds (Relu*, default 512_000) are new + runtime-configurable.
//! Both surfaces exposed at every tier: out-of-place `relu_*` and in-place
//! `relu_*_inplace`; Tier II serial re-exports for the MSTS tiled path.

pub mod fp32;
pub mod f16;
pub mod bf16;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::{relu as relu_f32_serial, relu_inplace as relu_f32_inplace_serial};
pub use f16::{relu as relu_f16_serial, relu_inplace as relu_f16_inplace_serial};
pub use bf16::{relu as relu_bf16_serial, relu_inplace as relu_bf16_inplace_serial};
pub use i8::{relu as relu_i8_serial, relu_inplace as relu_i8_inplace_serial};

macro_rules! tier3_unary {
    ($ty:ty, $thr:expr, $out:ident, $inp:ident, $ser_out:path, $ser_inp:path) => {
        pub fn $out(in_buf: &[$ty], out_buf: &mut [$ty]) {
            let t = thresholds::get($thr);
            let n = in_buf.len();
            if n < t {
                $ser_out(in_buf, out_buf);
            } else {
                out_buf.par_chunks_mut(t).enumerate().for_each(|(i, chunk)| {
                    let start = i * t;
                    let end = (start + t).min(n);
                    $ser_out(&in_buf[start..end], chunk);
                });
            }
        }
        pub fn $inp(buf: &mut [$ty]) {
            let t = thresholds::get($thr);
            if buf.len() < t {
                $ser_inp(buf);
            } else {
                buf.par_chunks_mut(t).for_each(|chunk| $ser_inp(chunk));
            }
        }
    };
}

tier3_unary!(f32, Threshold::ReluF32, relu_f32, relu_f32_inplace, fp32::relu, fp32::relu_inplace);
tier3_unary!(half::f16, Threshold::ReluF16, relu_f16, relu_f16_inplace, f16::relu, f16::relu_inplace);
tier3_unary!(half::bf16, Threshold::ReluBf16, relu_bf16, relu_bf16_inplace, bf16::relu, bf16::relu_inplace);
tier3_unary!(i8, Threshold::ReluI8, relu_i8, relu_i8_inplace, i8::relu, i8::relu_inplace);

#[cfg(test)]
mod parity {
    use super::*;
    crate::assert_parity_vs_legacy!(unary,
        relu_f32_vs_legacy, fp32::relu, crate::cpu_old::ops::unary::relu::relu_f32,
        f32, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(unary,
        relu_f16_vs_legacy, f16::relu, crate::cpu_old::ops::unary::relu::relu_f16,
        f16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(unary,
        relu_bf16_vs_legacy, bf16::relu, crate::cpu_old::ops::unary::relu::relu_bf16,
        bf16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(unary,
        relu_i8_vs_legacy, i8::relu, crate::cpu_old::ops::unary::relu::relu_i8,
        i8, [1, 15, 16, 17, 31, 32, 33, 1023, 65_536], 0.0);
}
