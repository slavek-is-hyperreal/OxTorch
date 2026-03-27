pub mod bf16;
pub mod fp32;

use rayon::prelude::*;

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
