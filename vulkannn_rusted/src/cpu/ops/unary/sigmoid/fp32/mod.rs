//! FP32 sigmoid — Tier II serial dispatcher. Reuses the exp cores; see
//! docs/kernel_specs/sigmoid_spec.md.

pub mod sigmoid_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod sigmoid_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod sigmoid_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod sigmoid_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod sigmoid_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod sigmoid_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn sigmoid(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { sigmoid_f32_avx512::sigmoid(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { sigmoid_f32_avx2::sigmoid(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { sigmoid_f32_avx1::sigmoid(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { sigmoid_f32_sse2::sigmoid(in_buf, out_buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { sigmoid_f32_neon::sigmoid(in_buf, out_buf) },
        _ => sigmoid_f32_scalar::sigmoid(in_buf, out_buf),
    }
}

pub fn sigmoid_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { sigmoid_f32_avx512::sigmoid_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { sigmoid_f32_avx2::sigmoid_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { sigmoid_f32_avx1::sigmoid_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { sigmoid_f32_sse2::sigmoid_inplace(buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { sigmoid_f32_neon::sigmoid_inplace(buf) },
        _ => sigmoid_f32_scalar::sigmoid_inplace(buf),
    }
}

#[cfg(test)]
mod oracle_test {
    use super::*;
    fn oracle(x: f32) -> f32 { (1.0f64 / (1.0 + (-(x as f64)).exp())) as f32 }
    fn ulp(a: f32, b: f32) -> f64 {
        if a == b { return 0.0; }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as f64
    }
    fn check(f: unsafe fn(&[f32], &mut [f32])) {
        let mut xs: Vec<f32> = (-40000..=40000).map(|i| i as f32 * 0.001).collect();
        xs.extend_from_slice(&[0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 100.0, -100.0]);
        let mut out = vec![0f32; xs.len()];
        unsafe { f(&xs, &mut out) };
        let mut m = 0.0f64;
        for (&x, &g) in xs.iter().zip(out.iter()) {
            if x.is_nan() { assert!(g.is_nan()); continue; }
            if x == f32::INFINITY { assert_eq!(g, 1.0); continue; }
            if x == f32::NEG_INFINITY { assert_eq!(g, 0.0); continue; }
            let u = ulp(g, oracle(x));
            if u > m { m = u; }
            assert!(u <= 2.0, "sigmoid ULP {u} > 2 at x={x} got={g} want={}", oracle(x));
        }
        eprintln!("sigmoid tier max ULP = {m}");
    }
    #[test] fn scalar() { check(sigmoid_f32_scalar::sigmoid); }
    #[cfg(target_arch = "x86_64")]
    #[test] fn sse2() { if is_x86_feature_detected!("sse2") { check(|i,o| unsafe { sigmoid_f32_sse2::sigmoid(i,o) }); } }
    #[cfg(target_arch = "x86_64")]
    #[test] fn avx1() { if is_x86_feature_detected!("avx") { check(|i,o| unsafe { sigmoid_f32_avx1::sigmoid(i,o) }); } }
}
