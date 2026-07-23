//! SILU — Tier III parallel gate. Rayon lives ONLY here. f16/bf16 use
//! convert-through-f32 (as legacy); i8 is a LUT. See docs/kernel_specs/silu_spec.md.

pub mod fp32;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::{silu as silu_f32_serial, silu_inplace as silu_f32_inplace_serial};
pub use i8::silu_i8;

pub fn silu_f32(in_buf: &[f32], out_buf: &mut [f32]) {
    let t = thresholds::get(Threshold::SiluF32);
    let n = in_buf.len();
    if n < t {
        fp32::silu(in_buf, out_buf);
    } else {
        out_buf.par_chunks_mut(t).enumerate().for_each(|(i, chunk)| {
            let s = i * t; let e = (s + t).min(n);
            fp32::silu(&in_buf[s..e], chunk);
        });
    }
}

pub fn silu_f32_inplace(buf: &mut [f32]) {
    let t = thresholds::get(Threshold::SiluF32);
    if buf.len() < t { fp32::silu_inplace(buf); }
    else { buf.par_chunks_mut(t).for_each(|c| fp32::silu_inplace(c)); }
}

macro_rules! half_silu {
    ($half:ty, $inp:ident, $out:ident, $ser:ident) => {
        pub fn $inp(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d, s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            silu_f32_inplace(&mut f);
            for (d, s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $out(in_buf: &[$half], out_buf: &mut [$half]) {
            let mut f = vec![0f32; in_buf.len()];
            for (d, s) in f.iter_mut().zip(in_buf.iter()) { *d = s.to_f32(); }
            silu_f32_inplace(&mut f);
            for (d, s) in out_buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $ser(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d, s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            fp32::silu_inplace(&mut f);
            for (d, s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
    };
}
half_silu!(half::f16, silu_f16_inplace, silu_f16, silu_f16_inplace_serial);
half_silu!(half::bf16, silu_bf16_inplace, silu_bf16, silu_bf16_inplace_serial);
