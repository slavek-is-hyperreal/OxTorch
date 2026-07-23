//! BF16 max — scalar (via f32). NaN ignored (legacy).
pub fn max(buf: &[half::bf16], initial: f32) -> f32 {
    buf.iter().fold(initial, |a, &b| a.max(b.to_f32()))
}
