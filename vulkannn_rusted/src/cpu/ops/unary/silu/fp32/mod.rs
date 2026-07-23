//! FP32 silu — Tier II serial dispatcher. Reuses exp cores; matches torch's
//! naive x/(1+exp(-x)) (incl. deep-tail flush). See docs/kernel_specs/silu_spec.md.

pub mod silu_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod silu_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod silu_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod silu_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod silu_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod silu_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn silu(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { silu_f32_avx512::silu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { silu_f32_avx2::silu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { silu_f32_avx1::silu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { silu_f32_sse2::silu(in_buf, out_buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { silu_f32_neon::silu(in_buf, out_buf) },
        _ => silu_f32_scalar::silu(in_buf, out_buf),
    }
}

pub fn silu_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { silu_f32_avx512::silu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { silu_f32_avx2::silu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { silu_f32_avx1::silu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { silu_f32_sse2::silu_inplace(buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { silu_f32_neon::silu_inplace(buf) },
        _ => silu_f32_scalar::silu_inplace(buf),
    }
}

#[cfg(test)]
mod oracle_test {
    use super::*;
    // f64 naive form over the well-conditioned range (|x|<=40); outside it torch
    // itself flushes and f64 is not the reference (see silu_spec.md).
    fn oracle(x: f32) -> f32 { ((x as f64) / (1.0 + (-(x as f64)).exp())) as f32 }
    fn ulp(a: f32, b: f32) -> f64 {
        if a == b { return 0.0; }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as f64
    }
    fn check(f: unsafe fn(&[f32], &mut [f32])) {
        let xs: Vec<f32> = (-40000..=40000).map(|i| i as f32 * 0.001).collect();
        let mut out = vec![0f32; xs.len()];
        unsafe { f(&xs, &mut out) };
        let mut m = 0.0f64;
        for (&x, &g) in xs.iter().zip(out.iter()) {
            let u = ulp(g, oracle(x));
            if u > m { m = u; }
            assert!(u <= 3.0, "silu ULP {u} > 3 at x={x} got={g} want={}", oracle(x));
        }
        // edges match torch
        let ed = [0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -100.0];
        let mut eo = vec![0f32; ed.len()];
        unsafe { f(&ed, &mut eo) };
        assert_eq!(eo[0], 0.0);
        assert_eq!(eo[1], f32::INFINITY);
        assert!(eo[2].is_nan(), "silu(-inf)=NaN (torch)");
        assert!(eo[3].is_nan());
        assert_eq!(eo[4], -0.0, "silu(-100)=-0.0 (torch flush)");
        eprintln!("silu tier max ULP = {m}");
    }
    #[test] fn scalar() { check(silu_f32_scalar::silu); }
    #[cfg(target_arch = "x86_64")]
    #[test] fn sse2() { if is_x86_feature_detected!("sse2") { check(|i,o| unsafe { silu_f32_sse2::silu(i,o) }); } }
    #[cfg(target_arch = "x86_64")]
    #[test] fn avx1() { if is_x86_feature_detected!("avx") { check(|i,o| unsafe { silu_f32_avx1::silu(i,o) }); } }
}
