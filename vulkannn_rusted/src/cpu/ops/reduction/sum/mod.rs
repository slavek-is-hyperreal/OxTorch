//! SUM — Tier III parallel reduction. Rayon lives ONLY here. The ACCUMULATOR is
//! always f64 (f32/f16/bf16) or i64 (i8); combining partial sums stays in f64/i64
//! and the downcast to the output dtype happens once, at the very end. See
//! fp32::sum_f32_scalar for the Rule-6 f64-accumulator policy.
//!
//! Two surfaces per dtype: the f64/i64 "raw accumulator" fn (for the MSTS tiled
//! path, which combines tile partials in f64) AND the downcast-to-output fn
//! (`sum_f32 -> f32`, `sum_i8 -> i64`) for flat callers.

pub mod fp32;
pub mod f16;
pub mod bf16;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

fn par_sum_f64<T: Sync>(buf: &[T], t: usize, serial: impl Fn(&[T]) -> f64 + Sync) -> f64 {
    if buf.len() <= t {
        serial(buf)
    } else {
        buf.par_chunks(t).map(|c| serial(c)).sum::<f64>()
    }
}

/// Raw f64 accumulator (for the MSTS tiled path). Serial Tier II.
pub fn sum_f32_acc(buf: &[f32]) -> f64 { fp32::sum(buf) }
pub fn sum_f16_acc(buf: &[half::f16]) -> f64 { f16::sum(buf) }
pub fn sum_bf16_acc(buf: &[half::bf16]) -> f64 { bf16::sum(buf) }
pub fn sum_i8_acc(buf: &[i8]) -> i64 { i8::sum(buf) }

/// Tier III parallel sum, downcast to output dtype at the end.
pub fn sum_f32(buf: &[f32]) -> f32 {
    par_sum_f64(buf, thresholds::get(Threshold::SumF32), fp32::sum) as f32
}
pub fn sum_f16(buf: &[half::f16]) -> f32 {
    par_sum_f64(buf, thresholds::get(Threshold::SumF16), f16::sum) as f32
}
pub fn sum_bf16(buf: &[half::bf16]) -> f32 {
    par_sum_f64(buf, thresholds::get(Threshold::SumBf16), bf16::sum) as f32
}
pub fn sum_i8(buf: &[i8]) -> i64 {
    let t = thresholds::get(Threshold::SumI8);
    if buf.len() <= t { i8::sum(buf) } else { buf.par_chunks(t).map(i8::sum).sum::<i64>() }
}
