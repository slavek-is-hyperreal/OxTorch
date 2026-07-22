pub mod bf16;
pub mod fp32;
pub mod f16;
pub mod i8;

use rayon::prelude::*;
use crate::cpu::thresholds::{self, Threshold};

// Tier II serial entries (arch-uniform) for the MSTS tiled path.
pub use f16::add_f16 as add_f16_serial;
pub use i8::add_i8 as add_i8_serial;

/// Threshold for using multiple threads.
/// Blocks below this size (like MSTS tiles) are processed serially to avoid context switches.
const PAR_THRESHOLD: usize = 512_000;

pub fn add_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    if n < PAR_THRESHOLD {
        bf16::add_bf16(a, b, res);
    } else {
        res.par_chunks_mut(PAR_THRESHOLD)
            .enumerate()
            .for_each(|(i, chunk_res)| {
                let start = i * PAR_THRESHOLD;
                let end = (start + PAR_THRESHOLD).min(n);
                bf16::add_bf16(&a[start..end], &b[start..end], chunk_res);
            });
    }
}

pub fn add_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    if n < PAR_THRESHOLD {
        // MSTS Tile / High-Performance Serial Matrix Path
        fp32::add(a, b, res);
    } else {
        // Parallel Core Path
        res.par_chunks_mut(PAR_THRESHOLD)
            .enumerate()
            .for_each(|(i, chunk_res)| {
                let start = i * PAR_THRESHOLD;
                let end = (start + PAR_THRESHOLD).min(n);
                fp32::add(&a[start..end], &b[start..end], chunk_res);
            });
    }
}

pub fn add_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let t = thresholds::get(Threshold::AddF16);
    let n = a.len();
    if n < t {
        f16::add_f16(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            f16::add_f16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

pub fn add_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    let t = thresholds::get(Threshold::AddI8);
    let n = a.len();
    if n < t {
        i8::add_i8(a, b, res);
    } else {
        res.par_chunks_mut(t).enumerate().for_each(|(i, chunk_res)| {
            let start = i * t;
            let end = (start + t).min(n);
            i8::add_i8(&a[start..end], &b[start..end], chunk_res);
        });
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    // f16: new Tier II serial vs legacy, bit-exact (same f16c/scalar math).
    crate::assert_parity_vs_legacy!(
        add_f16_vs_legacy, f16::add_f16, crate::cpu_old::ops::binary::add::add_f16,
        f16, [1, 7, 8, 9, 17, 1023, 1024, 1025, 65_536], 0.0);
    // i8: compared against the SCALAR reference (unambiguous saturating add),
    // NOT legacy — legacy's no-AVX2 path was the buggy u64 SWAR (see i8/mod.rs).
    crate::assert_parity_vs_legacy!(
        add_i8_vs_scalar, i8::add_i8, i8::add_i8_scalar::add,
        i8, [1, 15, 16, 17, 31, 32, 33, 1023, 65_536], 0.0);

    // Evidence for the deviation: legacy add_i8's u64 SWAR leaks carries across
    // byte lanes, so it disagrees with correct saturating add. This pins the bug
    // so the "not transcribed" decision is verifiable, not just asserted.
    #[test]
    fn legacy_add_i8_swar_is_buggy() {
        // byte0 overflows unsigned (0xC8+0xC8) and carries into byte1; byte1's
        // correct saturating result is 0, but the leaked carry makes legacy = 1.
        let a = [-56i8, 0, 0, 0, 0, 0, 0, 0]; // 0xC8, 0, ...
        let b = [-56i8, 0, 0, 0, 0, 0, 0, 0];
        let mut legacy = [0i8; 8];
        let mut correct = [0i8; 8];
        crate::cpu_old::ops::binary::add::add_i8(&a, &b, &mut legacy);
        i8::add_i8_scalar::add(&a, &b, &mut correct);
        assert_eq!(correct[1], 0, "saturating add: 0+0 = 0");
        assert_eq!(legacy[1], 1, "legacy SWAR leaks a carry into byte 1");
        assert_ne!(legacy, correct, "legacy diverges from correct saturating add");
    }
}
