//! BF16 sum — f64 accumulation (convert each bf16 -> f32 -> f64). Rule 6.
pub fn sum(buf: &[half::bf16]) -> f64 {
    let mut acc = 0.0f64;
    for &x in buf { acc += x.to_f32() as f64; }
    acc
}
