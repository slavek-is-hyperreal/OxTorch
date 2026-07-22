//! FP32 POW — Tier II serial dispatcher (tensor ^ scalar exponent).
//!
//! Two regimes (as legacy): exponent == 2.0 uses a SIMD square kernel selected
//! by `active_arch()`; every other exponent uses the scalar `powf` path (there
//! is no vectorised general pow in std::arch). Both surfaces exposed:
//! out-of-place `pow` and in-place `pow_inplace` (msts.rs / tensor ops need both).

pub mod pow_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod pow_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod pow_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod pow_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod pow_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod pow_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn pow(in_buf: &[f32], out_buf: &mut [f32], exponent: f32) {
    if exponent != 2.0 {
        return pow_f32_scalar::pow(in_buf, out_buf, exponent);
    }
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { pow_f32_avx512::square(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { pow_f32_avx2::square(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { pow_f32_avx1::square(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { pow_f32_sse2::square(in_buf, out_buf) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { pow_f32_neon::square(in_buf, out_buf) },

        _ => pow_f32_scalar::pow(in_buf, out_buf, 2.0),
    }
}

pub fn pow_inplace(buf: &mut [f32], exponent: f32) {
    if exponent != 2.0 {
        return pow_f32_scalar::pow_inplace(buf, exponent);
    }
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { pow_f32_avx512::square_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { pow_f32_avx2::square_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { pow_f32_avx1::square_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { pow_f32_sse2::square_inplace(buf) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { pow_f32_neon::square_inplace(buf) },

        _ => pow_f32_scalar::pow_inplace(buf, 2.0),
    }
}
