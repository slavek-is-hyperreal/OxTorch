//! EXP — Tier III parallel gate. Rayon lives ONLY here. f16/bf16 use
//! convert-through-f32 (whole buffer, as legacy). See docs/kernel_specs/exp_spec.md.
//! Both surfaces (out-of-place `exp_f32` / in-place `exp_f32_inplace`) + Tier II
//! serial re-exports for the MSTS tiled path.

pub mod fp32;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::{exp as exp_f32_serial, exp_inplace as exp_f32_inplace_serial};

pub fn exp_f32(in_buf: &[f32], out_buf: &mut [f32]) {
    let t = thresholds::get(Threshold::ExpF32);
    let n = in_buf.len();
    if n < t {
        fp32::exp(in_buf, out_buf);
    } else {
        out_buf.par_chunks_mut(t).enumerate().for_each(|(i, chunk)| {
            let start = i * t;
            let end = (start + t).min(n);
            fp32::exp(&in_buf[start..end], chunk);
        });
    }
}

pub fn exp_f32_inplace(buf: &mut [f32]) {
    let t = thresholds::get(Threshold::ExpF32);
    if buf.len() < t {
        fp32::exp_inplace(buf);
    } else {
        buf.par_chunks_mut(t).for_each(|chunk| fp32::exp_inplace(chunk));
    }
}

// f16/bf16: convert whole buffer through f32 (as legacy cpu_old/exp/mod.rs), then
// the f32 Tier III kernel, then convert back. In-place surface only (matches
// legacy + msts usage). Out-of-place provided for uniformity.
macro_rules! half_exp {
    ($half:ty, $inp:ident, $out:ident, $ser_inp:ident) => {
        pub fn $inp(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d, s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            exp_f32_inplace(&mut f);
            for (d, s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $out(in_buf: &[$half], out_buf: &mut [$half]) {
            let mut f = vec![0f32; in_buf.len()];
            for (d, s) in f.iter_mut().zip(in_buf.iter()) { *d = s.to_f32(); }
            exp_f32_inplace(&mut f);
            for (d, s) in out_buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        // Tier II serial (bypasses the f32 rayon gate for tiled callers).
        pub fn $ser_inp(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d, s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            fp32::exp_inplace(&mut f);
            for (d, s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
    };
}

half_exp!(half::f16, exp_f16_inplace, exp_f16, exp_f16_inplace_serial);
half_exp!(half::bf16, exp_bf16_inplace, exp_bf16, exp_bf16_inplace_serial);

#[cfg(test)]
mod parity {
    // f16/bf16 exp match legacy (both convert-through-f32; the f32 core differs
    // from legacy but for f16/bf16 the result rounds to the same half almost
    // everywhere — tol here is 1 half-ULP via bit compare would be too strict,
    // so we assert vs the f64 oracle rounded to the half type instead).
    fn oracle_half_f16(x: half::f16) -> half::f16 {
        half::f16::from_f32((x.to_f32() as f64).exp() as f32)
    }
    #[test]
    fn exp_f16_vs_oracle() {
        let xs: Vec<half::f16> = (-2000..=2000)
            .map(|i| half::f16::from_f32(i as f32 * 0.02))
            .collect();
        let mut got = xs.clone();
        super::exp_f16_inplace(&mut got);
        for (x, g) in xs.iter().zip(got.iter()) {
            let want = oracle_half_f16(*x);
            let (gf, wf) = (g.to_f32(), want.to_f32());
            if gf == wf || (gf.is_infinite() && wf.is_infinite()) { continue; }
            let d = (gf - wf).abs();
            let tol = wf.abs() * 1e-2 + 1e-3;
            assert!(d <= tol, "exp_f16 x={} got={} want={}", x.to_f32(), gf, wf);
        }
    }
}
