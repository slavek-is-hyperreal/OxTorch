//! I8 SUB — Tier II serial dispatcher (saturating).
//!
//! Tiers: scalar, sse2 (no-AVX x86), avx2, neon. Legacy's `swar` slot for sub_i8
//! already delegated to SSE2 on x86 (`_mm_subs_epi8`), so — unlike add_i8 —
//! there is no buggy u64 path here; this is a faithful transcription.

pub mod sub_i8_scalar;

#[cfg(target_arch = "x86_64")]
pub mod sub_i8_sse2;
#[cfg(target_arch = "x86_64")]
pub mod sub_i8_avx2;

#[cfg(target_arch = "aarch64")]
pub mod sub_i8_neon;

use crate::cpu::dispatch::Arch;

pub fn sub_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { sub_i8_avx2::sub_i8_avx2(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 | Arch::Avx1 | Arch::Sse2 => unsafe { sub_i8_sse2::sub_i8_sse2(a, b, res) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { sub_i8_neon::sub_i8_neon(a, b, res) },

        _ => sub_i8_scalar::sub(a, b, res),
    }
}
