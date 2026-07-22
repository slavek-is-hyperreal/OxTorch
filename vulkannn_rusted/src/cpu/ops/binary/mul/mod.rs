//! MUL — Tier III parallel gate. Rayon lives ONLY here; per-dtype/arch kernels
//! are serial. Thresholds are runtime-configurable (cpu::thresholds).
//! Both surfaces are exposed: the Tier III parallel fn AND the Tier II serial fn
//! (via `pub use`), so the MSTS tiled path can bypass rayon.

pub mod bf16;
pub mod fp32;
pub mod f16;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

// Re-export Tier II serial entries (arch-uniform) for the MSTS tiled path.
pub use fp32::mul as mul_f32_serial;
pub use f16::mul_f16 as mul_f16_serial;
pub use i8::mul_i8 as mul_i8_serial;

/// Specialized broadcast multiply: `[M, N] * [M, 1] -> [M, N]` (row-wise scale).
/// Tier III — owns its rayon over rows; per-row SIMD is `fp32::mul_broadcast_row`.
/// Transcribed from cpu_old/ops/binary/mul/mul_f32.rs::mul_broadcast_f32.
pub fn mul_broadcast_f32(a: &[f32], b: &[f32], res: &mut [f32], _m: usize, n: usize) {
    res.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * n..(i + 1) * n];
        fp32::mul_broadcast_row(a_row, b[i], row);
    });
}

const BF16_THRESHOLD: usize = 512_000;

pub fn mul_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    if n < BF16_THRESHOLD {
        bf16::mul_bf16(a, b, res);
    } else {
        res.par_chunks_mut(BF16_THRESHOLD).enumerate().for_each(|(i, chunk_res)| {
            let start = i * BF16_THRESHOLD;
            let end = (start + BF16_THRESHOLD).min(n);
            bf16::mul_bf16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn mul_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    let t = thresholds::get(Threshold::MulF32);
    let n = a.len();
    if n < t {
        fp32::mul(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            fp32::mul(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn mul_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let t = thresholds::get(Threshold::MulF16);
    let n = a.len();
    if n < t {
        f16::mul_f16(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            f16::mul_f16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn mul_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    let t = thresholds::get(Threshold::MulI8);
    let n = a.len();
    if n < t {
        i8::mul_i8(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            i8::mul_i8(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

// Parity vs legacy (Rule 1: legacy is source of truth). New Tier II serial
// kernels must match cpu_old bit-for-bit — same math, so tol = 0. Shapes stay
// below every legacy PAR_THRESHOLD so legacy also runs serial. Also writes
// golden snapshots that outlive cpu_old (deleted in wave 6).
#[cfg(test)]
mod parity {
    use super::*;
    crate::assert_parity_vs_legacy!(
        mul_f32_vs_legacy, fp32::mul, crate::cpu_old::ops::binary::mul::mul_f32,
        f32, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(
        mul_f16_vs_legacy, f16::mul_f16, crate::cpu_old::ops::binary::mul::mul_f16,
        f16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    crate::assert_parity_vs_legacy!(
        mul_i8_vs_legacy, i8::mul_i8, crate::cpu_old::ops::binary::mul::mul_i8,
        i8, [1, 15, 16, 17, 31, 32, 33, 1023, 65_536], 0.0);
}
