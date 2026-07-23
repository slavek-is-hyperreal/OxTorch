//! MAX — Tier III parallel reduction. Rayon lives ONLY here. Combines partial
//! maxes with `.max()` (NaN-ignoring, like legacy; docs/known_divergences §5).
pub mod fp32; pub mod f16; pub mod bf16; pub mod i8;
use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};
pub use fp32::max as max_f32_serial;
pub fn max_f32(buf: &[f32], initial: f32) -> f32 {
    let t = thresholds::get(Threshold::MaxF32);
    if buf.len() <= t { fp32::max(buf, initial) }
    else { buf.par_chunks(t).map(|c| fp32::max(c, initial)).reduce(|| initial, |a,b| a.max(b)) }
}
pub fn max_f16(buf: &[half::f16], initial: f32) -> f32 {
    let t = thresholds::get(Threshold::MaxF16);
    if buf.len() <= t { f16::max(buf, initial) }
    else { buf.par_chunks(t).map(|c| f16::max(c, initial)).reduce(|| initial, |a,b| a.max(b)) }
}
pub fn max_bf16(buf: &[half::bf16], initial: f32) -> f32 {
    let t = thresholds::get(Threshold::MaxBf16);
    if buf.len() <= t { bf16::max(buf, initial) }
    else { buf.par_chunks(t).map(|c| bf16::max(c, initial)).reduce(|| initial, |a,b| a.max(b)) }
}
pub fn max_i8(buf: &[i8], initial: i8) -> i8 {
    let t = thresholds::get(Threshold::MaxI8);
    if buf.len() <= t { i8::max(buf, initial) }
    else { buf.par_chunks(t).map(|c| i8::max(c, initial)).reduce(|| initial, |a,b| a.max(b)) }
}
