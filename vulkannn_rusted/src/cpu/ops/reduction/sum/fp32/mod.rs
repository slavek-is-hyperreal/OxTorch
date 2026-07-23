//! FP32 sum — Tier II serial reducer. Returns the f64 ACCUMULATOR (not f32); the
//! downcast to the output dtype happens once, at the Tier III / caller boundary,
//! so tiled callers can keep combining partials in f64. See sum_f32_scalar for
//! the f64-accumulator policy.

pub mod sum_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod sum_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod sum_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod sum_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod sum_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod sum_f32_neon;

use crate::cpu::dispatch::Arch;

/// Serial f32 sum with f64 accumulation. Returns f64 (downcast by the caller).
pub fn sum(buf: &[f32]) -> f64 {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { sum_f32_avx512::sum(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { sum_f32_avx2::sum(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { sum_f32_avx1::sum(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { sum_f32_sse2::sum(buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { sum_f32_neon::sum(buf) },
        _ => sum_f32_scalar::sum(buf),
    }
}

#[cfg(test)]
mod oracle_test {
    use super::*;
    // Oracle: strict left-to-right f64 sum (what the scalar tier computes).
    fn oracle(buf: &[f32]) -> f64 {
        let mut a = 0.0f64;
        for &x in buf { a += x as f64; }
        a
    }
    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n).map(|_| { s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 32.0 }).collect()
    }
    #[test]
    fn scalar_is_exact_f64() {
        for &n in &[0usize, 1, 7, 15, 16, 17, 1000, 1_000_000] {
            let v = data(n, n as u32 + 1);
            assert_eq!(sum_f32_scalar::sum(&v), oracle(&v));
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx1_matches_f64_within_1ulp() {
        if !is_x86_feature_detected!("avx") { return; }
        // SIMD reassociates the f64 adds, so it is NOT bit-equal to left-to-right;
        // both are valid f64 accumulations. Bound the disagreement tightly.
        for &n in &[16usize, 17, 31, 1000, 1_000_000] {
            let v = data(n, n as u32 + 7);
            let got = unsafe { sum_f32_avx1::sum(&v) };
            let want = oracle(&v);
            let rel = if want == 0.0 { got.abs() } else { ((got - want) / want).abs() };
            assert!(rel <= 1e-12, "avx1 sum rel err {rel} at n={n} (got {got}, want {want})");
        }
    }
}
