//! FP32 ATAN2 — Tier II serial dispatcher.
//!
//! Runtime feature detection via `cpu::dispatch::active_arch()` (honours the
//! `force_arch` override). SVE/SVE2/AVX-512 leaf kernels were removed in Wave 0
//! (never compiled on aarch64; unmeasured on x86). This surface is identical on
//! every architecture — arch differences live strictly in the leaf kernels.

pub mod atan2_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod atan2_f32_neon;

use crate::cpu::dispatch::Arch;

/// Dispatches Atan2 to the best available hardware kernel (or the forced arch).
pub fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { atan2_f32_avx512::atan2(y, x, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { atan2_f32_avx2::atan2(y, x, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { atan2_f32_avx1::atan2(y, x, res) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { atan2_f32_neon::atan2(y, x, res) },

        _ => atan2_f32_scalar::atan2(y, x, res),
    }
}
