pub mod bf16;
pub mod fp32;
pub mod f16;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

pub use f16::sub_f16 as sub_f16_serial;
pub use i8::sub_i8 as sub_i8_serial;

const PAR_THRESHOLD: usize = 512_000;

pub fn sub_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    if n < PAR_THRESHOLD {
        bf16::sub_bf16(a, b, res);
    } else {
        res.par_chunks_mut(PAR_THRESHOLD).enumerate().for_each(|(i, chunk_res)| {
            let start = i * PAR_THRESHOLD;
            let end = (start + PAR_THRESHOLD).min(n);
            bf16::sub_bf16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn sub_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    if n < PAR_THRESHOLD {
        fp32::sub(a, b, res);
    } else {
        res.par_chunks_mut(PAR_THRESHOLD).enumerate().for_each(|(i, chunk_res)| {
            let start = i * PAR_THRESHOLD;
            let end = (start + PAR_THRESHOLD).min(n);
            fp32::sub(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn sub_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let t = thresholds::get(Threshold::SubF16);
    let n = a.len();
    if n < t {
        f16::sub_f16(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            f16::sub_f16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn sub_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    let t = thresholds::get(Threshold::SubI8);
    let n = a.len();
    if n < t {
        i8::sub_i8(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            i8::sub_i8(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    crate::assert_parity_vs_legacy!(
        sub_f16_vs_legacy, f16::sub_f16, crate::cpu_old::ops::binary::sub::sub_f16,
        f16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    // sub_i8 legacy used correct SSE2 on x86 (no buggy swar), so vs-legacy holds.
    crate::assert_parity_vs_legacy!(
        sub_i8_vs_legacy, i8::sub_i8, crate::cpu_old::ops::binary::sub::sub_i8,
        i8, [1, 15, 16, 17, 31, 32, 33, 1023, 65_536], 0.0);
}
