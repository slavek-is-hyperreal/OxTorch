//! DIV — Tier III parallel gate. Rayon lives ONLY here. Thresholds are
//! runtime-configurable. Both surfaces exposed (Tier III parallel + Tier II
//! serial via `pub use`) so the MSTS tiled path can bypass rayon.
//! NOTE: legacy /0 -> 0.0 scalar quirk preserved (Rule 6; see fp32::div_f32_scalar).

pub mod bf16;
pub mod fp32;
pub mod f16;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::div as div_f32_serial;
pub use f16::div_f16 as div_f16_serial;
pub use i8::div_i8 as div_i8_serial;

const BF16_THRESHOLD: usize = 512_000;

pub fn div_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    if n < BF16_THRESHOLD {
        bf16::div_bf16(a, b, res);
    } else {
        res.par_chunks_mut(BF16_THRESHOLD).enumerate().for_each(|(i, chunk_res)| {
            let start = i * BF16_THRESHOLD;
            let end = (start + BF16_THRESHOLD).min(n);
            bf16::div_bf16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn div_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    let t = thresholds::get(Threshold::DivF32);
    let n = a.len();
    if n < t {
        fp32::div(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            fp32::div(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn div_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let t = thresholds::get(Threshold::DivF16);
    let n = a.len();
    if n < t {
        f16::div_f16(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            f16::div_f16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn div_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    let t = thresholds::get(Threshold::DivI8);
    let n = a.len();
    if n < t {
        i8::div_i8(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            i8::div_i8(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

// Parity vs legacy (Rule 1). New Tier II serial must match cpu_old bit-for-bit,
// INCLUDING the /0 quirk (scalar guards to 0.0; SIMD body divides raw). Shapes
// stay below every legacy PAR_THRESHOLD so legacy also runs serial. tol = 0.
#[cfg(test)]
mod parity {
    use super::*;
    crate::assert_parity_vs_legacy!(
        div_f32_vs_legacy, fp32::div, crate::cpu_old::ops::binary::div::div_f32,
        f32, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(
        div_f16_vs_legacy, f16::div_f16, crate::cpu_old::ops::binary::div::div_f16,
        f16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(
        div_i8_vs_legacy, i8::div_i8, crate::cpu_old::ops::binary::div::div_i8,
        i8, [1, 15, 16, 17, 31, 32, 33, 1023, 65_536], 0.0);
}
