//! I8 ADD — Tier II serial dispatcher (saturating).
//!
//! Tiers: scalar, sse2 (no-AVX x86), avx2, neon. `active_arch()` on an AVX1 CPU
//! reports Avx1, which maps here to sse2 (there is no avx1-specific i8 kernel and
//! `_mm_adds_epi8` is available whenever AVX1 is).
//!
//! NOTE (deliberate deviation from Rule 1): legacy add_i8 had a `swar` tier that
//! did a wrapping u64 add with per-lane overflow detection. That is BUGGY — a
//! plain u64 add carries across byte boundaries, so a byte with no signed
//! overflow can still be corrupted by a carry from the lane below (e.g.
//! a=[-56,0,..], b=[-56,0,..] yields byte1=1 instead of 0). It is not
//! transcribed. On this box legacy selected that buggy swar (no AVX2); the
//! migration routes to the correct sse2 kernel instead. A correct saturating i8
//! SWAR for GPR-only targets is a TODO, not a copied defect.

pub mod add_i8_scalar;

#[cfg(target_arch = "x86_64")]
pub mod add_i8_sse2;
#[cfg(target_arch = "x86_64")]
pub mod add_i8_avx2;

#[cfg(target_arch = "aarch64")]
pub mod add_i8_neon;

use crate::cpu::dispatch::Arch;

pub fn add_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { add_i8_avx2::add_i8_avx2(a, b, res) },
        // AVX-512/AVX1/SSE2 CPUs all have _mm_adds_epi8; no avx1/avx512-specific
        // i8 kernel exists, so they share the SSE2 kernel.
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 | Arch::Avx1 | Arch::Sse2 => unsafe { add_i8_sse2::add_i8_sse2(a, b, res) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { add_i8_neon::add_i8_neon(a, b, res) },

        _ => add_i8_scalar::add(a, b, res),
    }
}
