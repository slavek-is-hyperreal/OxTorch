//! POW — Tier III parallel gate (tensor ^ scalar exponent). Rayon lives ONLY
//! here. Legacy pow had no rayon; the threshold (PowF32, default 512_000) is new
//! and runtime-configurable. Both surfaces exposed: Tier III parallel + Tier II
//! serial (`pow_f32_serial` / `pow_f32_inplace_serial`) for the MSTS tiled path.

pub mod fp32;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::pow as pow_f32_serial;
pub use fp32::pow_inplace as pow_f32_inplace_serial;

pub fn pow_f32(in_buf: &[f32], out_buf: &mut [f32], exponent: f32) {
    let t = thresholds::get(Threshold::PowF32);
    let n = in_buf.len();
    if n < t {
        fp32::pow(in_buf, out_buf, exponent);
    } else {
        out_buf
            .par_chunks_mut(t)
            .enumerate()
            .for_each(|(i, chunk)| {
                let start = i * t;
                let end = (start + t).min(n);
                fp32::pow(&in_buf[start..end], chunk, exponent);
            });
    }
}

pub fn pow_f32_inplace(buf: &mut [f32], exponent: f32) {
    let t = thresholds::get(Threshold::PowF32);
    if buf.len() < t {
        fp32::pow_inplace(buf, exponent);
    } else {
        buf.par_chunks_mut(t).for_each(|chunk| fp32::pow_inplace(chunk, exponent));
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    // pow2 fast path: new Tier II serial vs legacy (bit-exact, x*x). Also covers
    // the general-exponent scalar path implicitly via the exponent!=2 branch.
    #[test]
    fn pow_f32_vs_legacy() {
        for &n in &[1usize, 7, 8, 9, 17, 1023, 1024, 1025, 65_536] {
            for &exp in &[2.0f32, 0.5, 3.0, -1.0] {
                let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.5 - 8.0)).collect();
                let mut got = vec![0f32; n];
                let mut want = vec![0f32; n];
                fp32::pow(&a, &mut got, exp);
                crate::cpu_old::ops::unary::pow::pow_f32(&a, &mut want, exp);
                // Bitwise: new and legacy run identical ops (x*x for exp==2, else
                // std powf), so bits match exactly — incl. NaN payloads, which
                // plain `==` would treat as unequal.
                let gb: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                assert_eq!(gb, wb, "pow mismatch n={n} exp={exp}");
            }
        }
    }
}
