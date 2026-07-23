//! FP32 tanh — Tier II serial dispatcher. Cephes two-branch on SIMD; see
//! docs/kernel_specs/tanh_spec.md. (Other tiers added after avx1 validates.)

pub mod tanh_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod tanh_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod tanh_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod tanh_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod tanh_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod tanh_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn tanh(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { tanh_f32_avx512::tanh(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { tanh_f32_avx2::tanh(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { tanh_f32_avx1::tanh(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { tanh_f32_sse2::tanh(in_buf, out_buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { tanh_f32_neon::tanh(in_buf, out_buf) },
        _ => tanh_f32_scalar::tanh(in_buf, out_buf),
    }
}

pub fn tanh_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { tanh_f32_avx512::tanh_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { tanh_f32_avx2::tanh_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { tanh_f32_avx1::tanh_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { tanh_f32_sse2::tanh_inplace(buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { tanh_f32_neon::tanh_inplace(buf) },
        _ => tanh_f32_scalar::tanh_inplace(buf),
    }
}

#[cfg(test)]
mod oracle_test {
    use super::*;
    fn oracle(x: f32) -> f32 { (x as f64).tanh() as f32 }
    fn ulp(a: f32, b: f32) -> f64 {
        if a == b { return 0.0; }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as f64
    }
    fn check(f: unsafe fn(&[f32], &mut [f32])) {
        let mut xs: Vec<f32> = (-100000..=100000).map(|i| i as f32 * 0.001).collect();
        xs.extend_from_slice(&[0.0, -0.0, 0.624, 0.625, 0.626, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 50.0, -50.0]);
        let mut out = vec![0f32; xs.len()];
        unsafe { f(&xs, &mut out) };
        let mut m = 0.0f64;
        for (&x, &g) in xs.iter().zip(out.iter()) {
            if x.is_nan() { assert!(g.is_nan()); continue; }
            if x == f32::INFINITY { assert_eq!(g, 1.0); continue; }
            if x == f32::NEG_INFINITY { assert_eq!(g, -1.0); continue; }
            let u = ulp(g, oracle(x));
            if u > m { m = u; }
            assert!(u <= 2.0, "tanh ULP {u} > 2 at x={x} got={g} want={}", oracle(x));
        }
        eprintln!("tanh tier max ULP = {m}");
    }
    #[test] fn scalar() { check(tanh_f32_scalar::tanh); }
    #[cfg(target_arch = "x86_64")]
    #[test] fn sse2() { if is_x86_feature_detected!("sse2") { check(|i,o| unsafe { tanh_f32_sse2::tanh(i,o) }); } }
    #[cfg(target_arch = "x86_64")]
    #[test] fn avx1() { if is_x86_feature_detected!("avx") { check(|i,o| unsafe { tanh_f32_avx1::tanh(i,o) }); } }
}
