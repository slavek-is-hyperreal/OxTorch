//! GELU — Tier III parallel gate (tanh-approx). f16/bf16 convert-through-f32;
//! i8 LUT. See docs/kernel_specs/gelu_spec.md.
pub mod fp32;
pub mod i8;
use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};
pub use fp32::{gelu as gelu_f32_serial, gelu_inplace as gelu_f32_inplace_serial};
pub use i8::gelu_i8;
pub fn gelu_f32(in_buf: &[f32], out_buf: &mut [f32]) {
    let t = thresholds::get(Threshold::GeluF32); let n = in_buf.len();
    if n < t { fp32::gelu(in_buf, out_buf); }
    else { out_buf.par_chunks_mut(t).enumerate().for_each(|(i,ch)| { let s=i*t; let e=(s+t).min(n); fp32::gelu(&in_buf[s..e], ch); }); }
}
pub fn gelu_f32_inplace(buf: &mut [f32]) {
    let t = thresholds::get(Threshold::GeluF32);
    if buf.len() < t { fp32::gelu_inplace(buf); } else { buf.par_chunks_mut(t).for_each(|c| fp32::gelu_inplace(c)); }
}
macro_rules! half_gelu {
    ($half:ty, $thr:expr, $inp:ident, $out:ident, $ser:ident) => {
        pub fn $inp(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d,s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            gelu_f32_inplace(&mut f);
            for (d,s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $out(in_buf: &[$half], out_buf: &mut [$half]) {
            let mut f = vec![0f32; in_buf.len()];
            for (d,s) in f.iter_mut().zip(in_buf.iter()) { *d = s.to_f32(); }
            gelu_f32_inplace(&mut f);
            for (d,s) in out_buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
        pub fn $ser(buf: &mut [$half]) {
            let mut f = vec![0f32; buf.len()];
            for (d,s) in f.iter_mut().zip(buf.iter()) { *d = s.to_f32(); }
            fp32::gelu_inplace(&mut f);
            for (d,s) in buf.iter_mut().zip(f.iter()) { *d = <$half>::from_f32(*s); }
        }
    };
}
half_gelu!(half::f16, Threshold::GeluF16, gelu_f16_inplace, gelu_f16, gelu_f16_inplace_serial);
half_gelu!(half::bf16, Threshold::GeluBf16, gelu_bf16_inplace, gelu_bf16, gelu_bf16_inplace_serial);
