//! NEG — Tier III parallel gate. Rayon lives ONLY here. Both surfaces + Tier II
//! serial re-exports for the MSTS tiled path. Thresholds Neg* (default 512_000).

pub mod fp32;
pub mod f16;
pub mod bf16;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::{neg as neg_f32_serial, neg_inplace as neg_f32_inplace_serial};
pub use f16::{neg as neg_f16_serial, neg_inplace as neg_f16_inplace_serial};
pub use bf16::{neg as neg_bf16_serial, neg_inplace as neg_bf16_inplace_serial};

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

tier3_unary!(f32, Threshold::NegF32, neg_f32, neg_f32_inplace, fp32::neg, fp32::neg_inplace);
tier3_unary!(half::f16, Threshold::NegF16, neg_f16, neg_f16_inplace, f16::neg, f16::neg_inplace);
tier3_unary!(half::bf16, Threshold::NegBf16, neg_bf16, neg_bf16_inplace, bf16::neg, bf16::neg_inplace);

#[cfg(test)]
mod parity {
    use super::*;
    crate::assert_parity_vs_legacy!(unary,
        neg_f32_vs_legacy, fp32::neg, crate::cpu_old::ops::unary::neg::neg_f32,
        f32, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(unary,
        neg_f16_vs_legacy, f16::neg, crate::cpu_old::ops::unary::neg::neg_f16,
        f16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(unary,
        neg_bf16_vs_legacy, bf16::neg, crate::cpu_old::ops::unary::neg::neg_bf16,
        bf16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
}
