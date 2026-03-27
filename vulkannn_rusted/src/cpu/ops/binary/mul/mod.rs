pub mod bf16;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 512_000;

pub fn mul_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n = a.len();
    if n < PAR_THRESHOLD {
        bf16::mul_bf16(a, b, res);
    } else {
        res.par_chunks_mut(PAR_THRESHOLD).enumerate().for_each(|(i, chunk_res)| {
            let start = i * PAR_THRESHOLD;
            let end = (start + PAR_THRESHOLD).min(n);
            bf16::mul_bf16(&a[start..end], &b[start..end], chunk_res);
        });
    }
}
