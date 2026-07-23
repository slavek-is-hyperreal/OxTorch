//! F16 sum — f64 accumulation (convert each half -> f32 -> f64). Rule 6 policy
//! (see fp32::sum_f32_scalar). Returns f64; caller downcasts.
pub fn sum(buf: &[half::f16]) -> f64 {
    let mut acc = 0.0f64;
    for &x in buf { acc += x.to_f32() as f64; }
    acc
}
