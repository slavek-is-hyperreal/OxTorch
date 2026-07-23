//! Per-token symmetric (absmax) quantization to Int8.
//!
//! MOVE-NOT-REWRITE (Wave 5): relocated verbatim from `cpu_old/ops/quantization.rs`.
//! Zero numeric changes (Rule 6): the seed `max_abs = 1e-7`, the `127.0/max_abs`
//! scale, the stored dequant scale `max_abs/127.0`, and the
//! `.round().clamp(-128.0, 127.0)` cast are all preserved bit-for-bit. The
//! per-token rayon loop stays here (this op has no arch/{dtype} tier split — the
//! AVX abs-max is an inline fast path inside the scalar reduction, exactly as
//! legacy had it). f16/bf16 upcast each element through f32 like legacy.

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Per-token symmetric quantization (absmax).
/// Returns Int8 tensor and F32 scales (1 per token).
pub fn quantize_per_token_absmax_f32(_m: usize, k: usize, src: &[f32], dst: &mut [i8], scales: &mut [f32]) {
    dst.par_chunks_mut(k).enumerate().zip(scales.par_iter_mut()).for_each(|((i, dst_row), scale_out)| {
        let src_row = &src[i * k .. (i + 1) * k];

        let mut max_abs = 1e-7f32;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    let mut max_v = _mm256_set1_ps(1e-7f32);
                    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

                    for chunk in src_row.chunks_exact(8) {
                        let v = _mm256_loadu_ps(chunk.as_ptr());
                        let abs_v = _mm256_and_ps(v, abs_mask);
                        max_v = _mm256_max_ps(max_v, abs_v);
                    }

                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
                    for &v in &tmp { if v > max_abs { max_abs = v; } }

                    // Remainder
                    let rem = (src_row.len() / 8) * 8;
                    for &v in &src_row[rem..] {
                        let abs = v.abs();
                        if abs > max_abs { max_abs = abs; }
                    }
                }
            } else {
                for &v in src_row {
                    let abs = v.abs();
                    if abs > max_abs { max_abs = abs; }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for &v in src_row {
                let abs = v.abs();
                if abs > max_abs { max_abs = abs; }
            }
        }

        let scale = 127.0 / max_abs;
        *scale_out = max_abs / 127.0; // Store the dequantization scale

        for j in 0..k {
            dst_row[j] = (src_row[j] * scale).round().clamp(-128.0, 127.0) as i8;
        }
    });
}

pub fn quantize_per_token_absmax_bf16(_m: usize, k: usize, src: &[half::bf16], dst: &mut [i8], scales: &mut [f32]) {
    dst.par_chunks_mut(k).enumerate().zip(scales.par_iter_mut()).for_each(|((i, dst_row), scale_out)| {
        let src_row = &src[i * k .. (i + 1) * k];
        let mut max_abs = 1e-7f32;

        for &v in src_row {
            let abs = v.to_f32().abs();
            if abs > max_abs { max_abs = abs; }
        }

        let scale = 127.0 / max_abs;
        *scale_out = max_abs / 127.0;

        for j in 0..k {
            dst_row[j] = (src_row[j].to_f32() * scale).round().clamp(-128.0, 127.0) as i8;
        }
    });
}

pub fn quantize_per_token_absmax_f16(_m: usize, k: usize, src: &[half::f16], dst: &mut [i8], scales: &mut [f32]) {
    dst.par_chunks_mut(k).enumerate().zip(scales.par_iter_mut()).for_each(|((i, dst_row), scale_out)| {
        let src_row = &src[i * k .. (i + 1) * k];
        let mut max_abs = 1e-7f32;

        for &v in src_row {
            let abs = v.to_f32().abs();
            if abs > max_abs { max_abs = abs; }
        }

        let scale = 127.0 / max_abs;
        *scale_out = max_abs / 127.0;

        for j in 0..k {
            dst_row[j] = (src_row[j].to_f32() * scale).round().clamp(-128.0, 127.0) as i8;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent scalar oracle for one row (no AVX, no rayon). This is the
    /// reference the migrated kernels must match bit-for-bit — for f32 it also
    /// pins the AVX abs-max fast path to the scalar semantics.
    fn oracle_row(row: &[f32]) -> (Vec<i8>, f32) {
        let mut max_abs = 1e-7f32;
        for &v in row {
            if v.abs() > max_abs { max_abs = v.abs(); }
        }
        let scale = 127.0 / max_abs;
        let out: Vec<i8> = row
            .iter()
            .map(|&v| (v * scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        (out, max_abs / 127.0)
    }

    // Rows chosen to exercise: len not a multiple of 8 (AVX remainder tail),
    // all-zeros (max_abs stays at the 1e-7 seed), saturation past ±127, a
    // negative-max row, and a plain multiple-of-8 row.
    fn cases() -> Vec<Vec<f32>> {
        vec![
            vec![0.0; 8],
            vec![0.0; 13],
            vec![1.0, -2.5, 3.3, -100.0, 100.0, 0.01, -0.01, 50.0, 7.0, -7.0, 0.0, 63.5, 127.9],
            vec![-9.0, -9.0, -9.0, 4.5, -1.0, 2.0, 8.0, -8.0],
            (0..17).map(|i| (i as f32 - 8.0) * 1.7).collect(),
            vec![1e30, -1e30, 3.0, 4.0], // huge -> scale tiny, everything saturates
        ]
    }

    #[test]
    fn f32_matches_scalar_oracle() {
        for row in cases() {
            let k = row.len();
            let mut dst = vec![0i8; k];
            let mut scales = vec![0f32; 1];
            quantize_per_token_absmax_f32(1, k, &row, &mut dst, &mut scales);
            let (want, want_scale) = oracle_row(&row);
            assert_eq!(dst, want, "dst mismatch for row {:?}", row);
            assert_eq!(scales[0].to_bits(), want_scale.to_bits(), "scale mismatch for row {:?}", row);
        }
    }

    #[test]
    fn f32_multi_row_independent_tokens() {
        // Two rows with different absmax must get independent scales.
        let row_a = vec![1.0, -2.0, 3.0, -0.5];
        let row_b = vec![10.0, -20.0, 5.0, 0.0];
        let mut src = row_a.clone();
        src.extend_from_slice(&row_b);
        let k = 4;
        let mut dst = vec![0i8; 2 * k];
        let mut scales = vec![0f32; 2];
        quantize_per_token_absmax_f32(2, k, &src, &mut dst, &mut scales);
        let (wa, sa) = oracle_row(&row_a);
        let (wb, sb) = oracle_row(&row_b);
        assert_eq!(&dst[..k], &wa[..]);
        assert_eq!(&dst[k..], &wb[..]);
        assert_eq!(scales[0].to_bits(), sa.to_bits());
        assert_eq!(scales[1].to_bits(), sb.to_bits());
    }

    #[test]
    fn f16_bf16_match_upcast_oracle() {
        for row in cases() {
            let k = row.len();
            // f16 / bf16 quantize the value AFTER a lossy round-trip through the
            // reduced dtype, so the oracle must round-trip too.
            let f16_row: Vec<half::f16> = row.iter().map(|&v| half::f16::from_f32(v)).collect();
            let bf16_row: Vec<half::bf16> = row.iter().map(|&v| half::bf16::from_f32(v)).collect();

            let f16_up: Vec<f32> = f16_row.iter().map(|v| v.to_f32()).collect();
            let bf16_up: Vec<f32> = bf16_row.iter().map(|v| v.to_f32()).collect();

            let mut dst = vec![0i8; k];
            let mut scales = vec![0f32; 1];
            quantize_per_token_absmax_f16(1, k, &f16_row, &mut dst, &mut scales);
            let (want, want_scale) = oracle_row(&f16_up);
            assert_eq!(dst, want, "f16 dst mismatch for {:?}", row);
            assert_eq!(scales[0].to_bits(), want_scale.to_bits());

            let mut dst = vec![0i8; k];
            let mut scales = vec![0f32; 1];
            quantize_per_token_absmax_bf16(1, k, &bf16_row, &mut dst, &mut scales);
            let (want, want_scale) = oracle_row(&bf16_up);
            assert_eq!(dst, want, "bf16 dst mismatch for {:?}", row);
            assert_eq!(scales[0].to_bits(), want_scale.to_bits());
        }
    }
}
