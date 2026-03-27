//! ATAN2 Binary Operation Module.
//! Scientific-Grade Register-Blocked Matrix.

pub mod fp32;

/// Parallel entry point for FP32 Atan2.
/// Dispatches to the 8-kernel specialization matrix.
pub fn atan2_f32(y: &[f32], x: &[f32], res: &mut [f32]) {
    use rayon::prelude::*;
    
    // Threshold for parallel processing (16k elements)
    if y.len() < 16384 {
        return fp32::atan2(y, x, res);
    }

    // Split into chunks for parallel execution
    let chunk_size = 8192;
    res.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(y.len());
            fp32::atan2(&y[start..end], &x[start..end], chunk);
        });
}
