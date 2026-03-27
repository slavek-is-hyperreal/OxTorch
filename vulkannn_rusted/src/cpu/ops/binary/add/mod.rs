pub mod bf16;
use rayon::prelude::*;

/// Threshold for using multiple threads. 
/// Blocks below this size (like MSTS tiles) are processed serially to avoid context switches.
const PAR_THRESHOLD: usize = 512_000;

pub fn add_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    if n < PAR_THRESHOLD {
        // Serial path: used for Direct ops and MSTS/Streaming tiles.
        bf16::add_bf16(a, b, res);
    } else {
        // Parallel path: used for large RAM-resident tensors.
        res.par_chunks_mut(PAR_THRESHOLD)
            .enumerate()
            .for_each(|(i, chunk_res)| {
                let start = i * PAR_THRESHOLD;
                let end = (start + PAR_THRESHOLD).min(n);
                bf16::add_bf16(&a[start..end], &b[start..end], chunk_res);
            });
    }
}
