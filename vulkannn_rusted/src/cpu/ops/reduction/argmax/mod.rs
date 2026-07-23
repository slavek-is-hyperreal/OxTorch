//! ARGMAX — axis-wise argmax reduction. Rayon over the outer*inner rows (Tier
//! III lives here). Scalar per-row scan (index tracking is data-dependent; §8 —
//! legacy has no SIMD argmax, transcribed as scalar). Only f32/f16 (legacy set).
//!
//! Tie-breaking: `val > max_val` (strict) keeps the FIRST max index — matches
//! torch. NaN: skipped (`NaN > x` is false), so argmax ignores NaN and picks the
//! max of the non-NaN values; torch returns the NaN index. Documented divergence
//! (docs/known_divergences.md §6), transcribed under Rule 6.

use rayon::prelude::*;

/// Serial per-row argmax over a strided view: elements at
/// `base + j*inner` for j in 0..dim_size. Returns the first-max index.
#[inline]
fn argmax_row_f32(base: &[f32], dim_size: usize, inner: usize) -> usize {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0usize;
    for j in 0..dim_size {
        let v = base[j * inner];
        if v > max_val {
            max_val = v;
            max_idx = j;
        }
    }
    max_idx
}

pub fn argmax_f32(in_buf: &[f32], out_buf: &mut [f32], _outer: usize, dim_size: usize, inner: usize) {
    out_buf.par_chunks_mut(inner).enumerate().for_each(|(i, out_row)| {
        for k in 0..inner {
            out_row[k] = argmax_row_f32(&in_buf[i * dim_size * inner + k..], dim_size, inner) as f32;
        }
    });
}

pub fn argmax_f16(in_buf: &[half::f16], out_buf: &mut [f32], _outer: usize, dim_size: usize, inner: usize) {
    out_buf.par_chunks_mut(inner).enumerate().for_each(|(i, out_row)| {
        for k in 0..inner {
            let base = i * dim_size * inner + k;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;
            for j in 0..dim_size {
                let v = in_buf[base + j * inner].to_f32();
                if v > max_val { max_val = v; max_idx = j; }
            }
            out_row[k] = max_idx as f32;
        }
    });
}

#[cfg(test)]
mod t {
    use super::*;
    #[test]
    fn contiguous_and_tie_and_nan() {
        // 1 row, dim=4, inner=1 (contiguous).
        let mut out = [0f32; 1];
        argmax_f32(&[3.0, 1.0, 3.0, 3.0], &mut out, 1, 4, 1);
        assert_eq!(out[0], 0.0, "tie -> first index (matches torch)");
        argmax_f32(&[1.0, f32::NAN, 3.0], &mut { let mut o=[0f32;1]; o }, 1, 3, 1);
        let mut o2 = [0f32; 1];
        argmax_f32(&[1.0, f32::NAN, 3.0], &mut o2, 1, 3, 1);
        assert_eq!(o2[0], 2.0, "NaN ignored -> index of 3 (diverges from torch=1)");
    }
    #[test]
    fn strided_2rows() {
        // 2 rows x dim 3, inner=1: [[1,5,2],[9,0,4]] -> [1, 0]
        let mut out = [0f32; 2];
        argmax_f32(&[1.0, 5.0, 2.0, 9.0, 0.0, 4.0], &mut out, 2, 3, 1);
        assert_eq!(out, [1.0, 0.0]);
    }
}
