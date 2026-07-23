//! Baseline Scalar f32 sum — NAIVE f64 ACCUMULATION.
//!
//! Rule 6 + Wave-3 policy: the accumulator is ALWAYS f64, regardless of input
//! dtype, and only downcast to the output type at the very end. Do NOT "optimise"
//! this to an f32 accumulator later — for a large tensor an f32 accumulator loses
//! precision / saturates. (Legacy's SIMD path accumulated in f32 registers; this
//! module deliberately corrects that to f64 throughout.) Naive f64 first — no
//! Kahan/pairwise unless a benchmark proves the need.

#[inline(always)]
pub fn sum(buf: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &x in buf {
        acc += x as f64;
    }
    acc
}
