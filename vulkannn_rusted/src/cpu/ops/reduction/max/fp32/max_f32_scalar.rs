//! Baseline Scalar f32 max. NOTE: `f32::max` ignores NaN (returns the non-NaN
//! operand) — legacy behaviour, transcribed; diverges from torch which
//! propagates NaN (docs/known_divergences.md §5).
#[inline(always)]
pub fn max(buf: &[f32], initial: f32) -> f32 {
    buf.iter().fold(initial, |a, &b| a.max(b))
}
