//! TANH — Tier III parallel gate. Rayon lives ONLY here. f16/bf16 use
//! convert-through-f32 (as legacy); i8 is a LUT. See docs/kernel_specs/tanh_spec.md.

pub mod fp32;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use fp32::{tanh as tanh_f32_serial, tanh_inplace as tanh_f32_inplace_serial};
pub use i8::tanh_i8;

pub fn tanh_f32(in_buf: &[f32], out_buf: &mut [f32]) {
    let t = thresholds::get(Threshold::TanhF32);
    let n = in_buf.len();
    if n < t {
        fp32::tanh(in_buf, out_buf);
    } else {
        out_buf.par_chunks_mut(t).enumerate().for_each(|(i, chunk)| {
            let s = i * t; let e = (s + t).min(n);
            fp32::tanh(&in_buf[s..e], chunk);
        });
    }
}

pub fn tanh_f32_inplace(buf: &mut [f32]) {
    let t = thresholds::get(Threshold::TanhF32);
    if buf.len() < t { fp32::tanh_inplace(buf); }
    else { buf.par_chunks_mut(t).for_each(|c| fp32::tanh_inplace(c)); }
}

macro_rules! half_tanh {
    ($half:ty, $inp:ident, $out:ident, $ser:ident) => {
        pub fn $inp(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d, s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            tanh_f32_inplace(&mut f);
            for (d, s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $out(in_buf: &[$half], out_buf: &mut [$half]) {
            let mut f = vec![0f32; in_buf.len()];
            for (d, s) in f.iter_mut().zip(in_buf.iter()) { *d = s.to_f32(); }
            tanh_f32_inplace(&mut f);
            for (d, s) in out_buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $ser(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d, s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            fp32::tanh_inplace(&mut f);
            for (d, s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
    };
}
half_tanh!(half::f16, tanh_f16_inplace, tanh_f16, tanh_f16_inplace_serial);
half_tanh!(half::bf16, tanh_bf16_inplace, tanh_bf16, tanh_bf16_inplace_serial);
