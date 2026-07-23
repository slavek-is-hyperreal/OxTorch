//! I8 sum — EXACT i64 accumulation (integer, no precision loss). Matches legacy
//! sum_i8 -> i64. (The f64-accumulator policy is for float dtypes; i8 sums are
//! exact in i64.)
pub fn sum(buf: &[i8]) -> i64 {
    let mut acc = 0i64;
    for &x in buf { acc += x as i64; }
    acc
}
