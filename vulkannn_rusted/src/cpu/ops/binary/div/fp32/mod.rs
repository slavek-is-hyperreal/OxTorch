//! FP32 DIV — Tier II serial dispatcher.
//!
//! Runtime feature detection via `cpu::dispatch::active_arch()` (honours the
//! `force_arch` override). Arch-uniform public surface; leaf kernels are serial.
//! div uses plain cached stores — no NON_TEMPORAL size gate.
//! See div_f32_scalar for the legacy /0-returns-0.0 quirk (preserved, Rule 6).

pub mod div_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod div_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod div_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod div_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod div_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod div_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn div(a: &[f32], b: &[f32], res: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { div_f32_avx512::div_f32_avx512(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { div_f32_avx2::div_f32_avx2(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { div_f32_avx1::div_f32_avx1(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { div_f32_sse2::div_f32_sse2(a, b, res) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { div_f32_neon::div_f32_neon(a, b, res) },

        _ => div_f32_scalar::div(a, b, res),
    }
}
