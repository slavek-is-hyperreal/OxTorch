//! FP32 ReLU — Tier II serial dispatcher. Both surfaces (out-of-place `relu` and
//! in-place `relu_inplace`) are exposed — msts.rs unary paths use in-place, other
//! callers use out-of-place. Runtime dispatch via `active_arch()`.

pub mod relu_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod relu_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod relu_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod relu_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod relu_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod relu_f32_neon;

use crate::cpu::dispatch::Arch;

pub fn relu(in_buf: &[f32], out_buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { relu_f32_avx512::relu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { relu_f32_avx2::relu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { relu_f32_avx1::relu(in_buf, out_buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { relu_f32_sse2::relu(in_buf, out_buf) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { relu_f32_neon::relu(in_buf, out_buf) },

        _ => relu_f32_scalar::relu(in_buf, out_buf),
    }
}

pub fn relu_inplace(buf: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { relu_f32_avx512::relu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { relu_f32_avx2::relu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { relu_f32_avx1::relu_inplace(buf) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { relu_f32_sse2::relu_inplace(buf) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { relu_f32_neon::relu_inplace(buf) },

        _ => relu_f32_scalar::relu_inplace(buf),
    }
}
