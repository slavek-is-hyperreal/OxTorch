//! FP32 exp — Tier II serial dispatcher. See docs/kernel_specs/exp_spec.md.
//! (SIMD tiers added incrementally after the avx1 kernel is validated vs oracle.)

pub mod exp_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod exp_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod exp_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod exp_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod exp_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod exp_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { exp_f32_avx512::exp(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { exp_f32_avx2::exp(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { exp_f32_avx1::exp(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { exp_f32_sse2::exp(in_buf, out_buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { exp_f32_neon::exp(in_buf, out_buf) },
        _ => exp_f32_scalar::exp(in_buf, out_buf),
    }
}

pub fn exp_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { exp_f32_avx512::exp_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { exp_f32_avx2::exp_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { exp_f32_avx1::exp_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { exp_f32_sse2::exp_inplace(buf) },
        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { exp_f32_neon::exp_inplace(buf) },
        _ => exp_f32_scalar::exp_inplace(buf),
    }
}

#[cfg(test)]
mod oracle_test {
    use super::*;

    // Oracle: exp in f64, rounded to f32 (docs/kernel_specs/README.md §1).
    fn oracle(x: f32) -> f32 {
        (x as f64).exp() as f32
    }
    // exp(x) > 0 for all finite x, so got/want are positive floats and adjacent
    // representable values differ by 1 in raw bits — ULP = |bits_a - bits_b|.
    fn ulp_diff(a: f32, b: f32) -> f64 {
        if a == b { return 0.0; }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as f64
    }

    fn check_tier(f: unsafe fn(&[f32], &mut [f32])) {
        // Dense sweep of the valid domain.
        let mut xs: Vec<f32> = Vec::new();
        let (lo, hi) = (-103.0f32, 88.7f32);
        let steps = 20000;
        for i in 0..=steps {
            xs.push(lo + (hi - lo) * (i as f32) / (steps as f32));
        }
        // Edge cases (exact match).
        xs.extend_from_slice(&[
            0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            88.72283905206835, 89.0, -103.3, -104.0, 700.0, -700.0,
        ]);
        let mut out = vec![0f32; xs.len()];
        unsafe { f(&xs, &mut out) };
        let mut max_ulp = 0.0f64;
        for (i, (&x, &got)) in xs.iter().zip(out.iter()).enumerate() {
            // Unambiguous edge cases: exact match.
            if x.is_nan() { assert!(got.is_nan(), "exp(NaN) must be NaN"); continue; }
            if x == f32::INFINITY { assert_eq!(got, f32::INFINITY, "exp(+inf)=+inf"); continue; }
            if x == f32::NEG_INFINITY { assert_eq!(got, 0.0, "exp(-inf)=0"); continue; }
            // Everything else — including the denormal band and the near-under/
            // overflow boundaries — is compared in ULP vs the f64 oracle. The
            // kernel flushing the smallest denormals to 0 costs <=1 ULP in raw
            // bits, which the >=2 bound absorbs.
            let want = oracle(x);
            let u = ulp_diff(got, want);
            if u > max_ulp { max_ulp = u; }
            assert!(u <= 2.0, "exp ULP {u} > 2 at x={x} (got {got:e}, want {want:e}, idx {i})");
        }
        assert!(max_ulp <= 2.0, "max ULP {max_ulp} exceeds bound");
        eprintln!("exp tier max ULP = {max_ulp}");
    }

    #[test]
    fn scalar_within_2ulp() { check_tier(exp_f32_scalar::exp); }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_within_2ulp() {
        if is_x86_feature_detected!("sse2") {
            check_tier(|i, o| unsafe { exp_f32_sse2::exp(i, o) });
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx1_within_2ulp() {
        if is_x86_feature_detected!("avx") {
            check_tier(|i, o| unsafe { exp_f32_avx1::exp(i, o) });
        }
    }
}
