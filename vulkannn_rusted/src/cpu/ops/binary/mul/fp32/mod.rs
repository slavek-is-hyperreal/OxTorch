//! FP32 MUL — Tier II serial dispatcher.
//!
//! Runtime feature detection via `cpu::dispatch::active_arch()` (honours the
//! `force_arch` override). Tier II contract: this public surface is identical on
//! every architecture; arch differences live strictly in the leaf kernels.
//! mul uses plain (cached) stores — no NON_TEMPORAL size gate needed (unlike add).

pub mod mul_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod mul_f32_sse2;
#[cfg(target_arch = "x86_64")]
pub mod mul_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod mul_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod mul_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod mul_f32_neon;

use crate::cpu::dispatch::Arch;

/// Dispatches FP32 MUL to the best kernel the CPU supports (or the forced arch).
pub fn mul(a: &[f32], b: &[f32], res: &mut [f32]) {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 => unsafe { mul_f32_avx512::mul_f32_avx512(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 => unsafe { mul_f32_avx2::mul_f32_avx2(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 => unsafe { mul_f32_avx1::mul_f32_avx1(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Sse2 => unsafe { mul_f32_sse2::mul_f32_sse2(a, b, res) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { mul_f32_neon::mul_f32_neon(a, b, res) },

        _ => mul_f32_scalar::mul(a, b, res),
    }
}

/// Serial single-row broadcast scale: `row[j] = a_row[j] * scale`. The per-row
/// SIMD kernel for `mul_broadcast_f32`; the rayon-over-rows wrapper lives at the
/// Tier III level in `mul/mod.rs` (rayon is not permitted below `{op}/mod.rs`).
pub fn mul_broadcast_row(a_row: &[f32], scale: f32, row: &mut [f32]) {
    let n = a_row.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe {
                use std::arch::x86_64::*;
                let v_scale = _mm256_set1_ps(scale);
                let n8 = (n / 8) * 8;
                for j in (0..n8).step_by(8) {
                    let va = _mm256_loadu_ps(a_row.as_ptr().add(j));
                    _mm256_storeu_ps(row.as_mut_ptr().add(j), _mm256_mul_ps(va, v_scale));
                }
                for j in n8..n {
                    row[j] = a_row[j] * scale;
                }
                return;
            }
        }
    }
    for j in 0..n {
        row[j] = a_row[j] * scale;
    }
}
