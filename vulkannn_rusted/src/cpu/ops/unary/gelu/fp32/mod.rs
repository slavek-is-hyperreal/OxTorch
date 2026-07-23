//! FP32 gelu (tanh-approx) — Tier II serial dispatcher. Reuses tanh cores.
//! See docs/kernel_specs/gelu_spec.md. (Other tiers added after avx1 validates.)

pub mod gelu_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod gelu_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod gelu_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod gelu_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod gelu_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod gelu_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn gelu(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { gelu_f32_avx512::gelu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { gelu_f32_avx2::gelu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { gelu_f32_avx1::gelu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { gelu_f32_sse2::gelu(in_buf, out_buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { gelu_f32_neon::gelu(in_buf, out_buf) },
        _ => gelu_f32_scalar::gelu(in_buf, out_buf),
    }
}

pub fn gelu_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { gelu_f32_avx512::gelu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { gelu_f32_avx2::gelu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { gelu_f32_avx1::gelu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { gelu_f32_sse2::gelu_inplace(buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { gelu_f32_neon::gelu_inplace(buf) },
        _ => gelu_f32_scalar::gelu_inplace(buf),
    }
}

#[cfg(test)]
mod oracle_test {
    use super::*;
    fn oracle(x: f32) -> f32 {
        let (k, c) = (0.7978845608f64, 0.044715f64);
        let xd = x as f64;
        (0.5 * xd * (1.0 + (k * (xd + c * xd * xd * xd)).tanh())) as f32
    }
    fn ulp(a: f32, b: f32) -> f64 {
        if a == b { return 0.0; }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as f64
    }
    fn check(f: unsafe fn(&[f32], &mut [f32])) {
        // Combined bound (spec: "<=4 ULP OR atol 1e-6"). Bit-ULP alone is too
        // harsh near zero: f32 tanh saturates to exactly ±1 for |inner| > ~9 (so
        // gelu flushes to ±0) while the f64 oracle keeps a ~1e-11..1e-16 tail;
        // torch (f32) flushes the same way, so an absolute floor is the right
        // metric there. Pass if within 4 ULP OR within atol 1e-6.
        let xs: Vec<f32> = (-40000..=40000).map(|i| i as f32 * 0.001).collect();
        let mut out = vec![0f32; xs.len()];
        unsafe { f(&xs, &mut out) };
        let mut m = 0.0f64;
        for (&x, &g) in xs.iter().zip(out.iter()) {
            let want = oracle(x);
            let u = ulp(g, want);
            let abs = (g as f64 - want as f64).abs();
            if u < m.max(0.0) || u <= 4.0 { if u > m { m = u; } }
            assert!(u <= 4.0 || abs <= 1e-6, "gelu ULP {u} abs {abs} at x={x} got={g} want={want}");
        }
        // edges match torch approximate='tanh'
        let ed = [0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -100.0];
        let mut eo = vec![0f32; ed.len()];
        unsafe { f(&ed, &mut eo) };
        assert_eq!(eo[0], 0.0);
        assert_eq!(eo[1], f32::INFINITY);
        assert!(eo[2].is_nan(), "gelu(-inf)=NaN (torch tanh)");
        assert!(eo[3].is_nan());
        assert_eq!(eo[4], 0.0, "gelu(-100)=-0.0 (torch flush)");
        eprintln!("gelu tier max ULP = {m}");
    }
    #[test] fn scalar() { check(gelu_f32_scalar::gelu); }
    #[cfg(target_arch = "x86_64")]
    #[test] fn sse2() { if is_x86_feature_detected!("sse2") { check(|i,o| unsafe { gelu_f32_sse2::gelu(i,o) }); } }
    #[cfg(target_arch = "x86_64")]
    #[test] fn avx1() { if is_x86_feature_detected!("avx") { check(|i,o| unsafe { gelu_f32_avx1::gelu(i,o) }); } }
}
