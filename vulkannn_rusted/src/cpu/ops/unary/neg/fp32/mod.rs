//! FP32 negation — Tier II serial dispatcher. Both surfaces (out-of-place `neg`
//! and in-place `neg_inplace`) via `active_arch()`.

pub mod neg_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod neg_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod neg_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod neg_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod neg_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod neg_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn neg(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { neg_f32_avx512::neg(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { neg_f32_avx2::neg(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { neg_f32_avx1::neg(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { neg_f32_sse2::neg(in_buf, out_buf) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { neg_f32_neon::neg(in_buf, out_buf) },

        _ => neg_f32_scalar::neg(in_buf, out_buf),
    }
}

pub fn neg_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { neg_f32_avx512::neg_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { neg_f32_avx2::neg_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { neg_f32_avx1::neg_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { neg_f32_sse2::neg_inplace(buf) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { neg_f32_neon::neg_inplace(buf) },

        _ => neg_f32_scalar::neg_inplace(buf),
    }
}
