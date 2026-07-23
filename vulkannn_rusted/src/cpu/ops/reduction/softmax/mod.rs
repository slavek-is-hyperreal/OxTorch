//! SOFTMAX — per-row softmax / log-softmax (in-place). The tensor layer
//! parallelises over rows, so this is the serial per-row op (as legacy).
//!
//! Reuses the validated Wave-2/3 cores: `max` (numerically-stable subtraction),
//! the `exp` core (<=2 ULP), and the `sum` f64 ACCUMULATOR for the denominator.
//! Legacy summed the denominator in f32; using f64 here is MORE accurate than
//! torch (which accumulates in ~f32), so parity tolerance vs torch is looser, not
//! tighter. NaN propagates through the f64 sum -> all-NaN row, matching torch
//! (even though `max` itself ignores NaN — see known_divergences.md §5).

use crate::cpu::ops::reduction::{max, sum};
use crate::cpu::ops::unary::exp;

/// Core: softmax over a contiguous f32 row, in place.
pub fn softmax_f32(buf: &mut [f32], is_log: bool) {
    if buf.is_empty() { return; }
    let m = max::fp32::max(buf, f32::NEG_INFINITY);   // stabilisation (reuse max)
    for x in buf.iter_mut() { *x -= m; }              // buf = x - max
    let mut e = vec![0f32; buf.len()];
    exp::fp32::exp(buf, &mut e);                       // e = exp(x - max) (reuse exp core)
    let denom = sum::fp32::sum(&e);                    // f64 accumulator (reuse sum)
    if is_log {
        let log_sum = denom.ln() as f32;              // ln in f64, downcast once
        for x in buf.iter_mut() { *x -= log_sum; }    // (x - max) - log_sum
    } else {
        let inv = (1.0 / denom) as f32;
        for (x, &ev) in buf.iter_mut().zip(e.iter()) { *x = ev * inv; }
    }
}

pub fn softmax_f16(buf: &mut [half::f16], is_log: bool) {
    if buf.is_empty() { return; }
    let mut f: Vec<f32> = buf.iter().map(|x| x.to_f32()).collect();
    softmax_f32(&mut f, is_log);
    for (d, &s) in buf.iter_mut().zip(f.iter()) { *d = half::f16::from_f32(s); }
}

pub fn softmax_bf16(buf: &mut [half::bf16], is_log: bool) {
    if buf.is_empty() { return; }
    let mut f: Vec<f32> = buf.iter().map(|x| x.to_f32()).collect();
    softmax_f32(&mut f, is_log);
    for (d, &s) in buf.iter_mut().zip(f.iter()) { *d = half::bf16::from_f32(s); }
}

pub fn softmax_i8(buf: &mut [i8], is_log: bool) {
    if buf.is_empty() { return; }
    let mut f: Vec<f32> = buf.iter().map(|&x| x as f32).collect();
    softmax_f32(&mut f, is_log);
    for (d, &s) in buf.iter_mut().zip(f.iter()) { *d = s.clamp(-128.0, 127.0) as i8; }
}

#[cfg(test)]
mod t {
    use super::*;
    fn oracle(row: &[f32]) -> Vec<f32> {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = row.iter().map(|&x| ((x - m) as f64).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|&v| (v / s) as f32).collect()
    }
    #[test]
    fn softmax_sums_to_one_and_matches_oracle() {
        let mut row = vec![1.0f32, 2.0, 3.0, -1.0, 0.5];
        let want = oracle(&row);
        softmax_f32(&mut row, false);
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "softmax sums to 1, got {s}");
        for (g, w) in row.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 4e-7, "got {g} want {w}");
        }
    }
    #[test]
    fn log_softmax_matches() {
        let mut row = vec![1.0f32, 2.0, 3.0];
        let mut lin = row.clone();
        softmax_f32(&mut lin, false);
        softmax_f32(&mut row, true);
        for (lg, l) in row.iter().zip(lin.iter()) {
            assert!((lg - l.ln()).abs() <= 1e-5, "log-softmax {lg} vs ln(softmax) {}", l.ln());
        }
    }
}
