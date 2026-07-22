//! FP32 ADD — Tier II serial dispatcher.
//!
//! Runtime feature detection via `cpu::dispatch::active_arch()`, which also
//! honours the `force_arch` override (debug / benchmarking). This replaces the
//! former compile-time `#[cfg(target_feature)]` ladder, which silently fell
//! through to scalar on any stock `maturin develop --release` build (no
//! RUSTFLAGS ⇒ no `target_feature` ⇒ every SIMD arm dead).
//!
//! Tier II contract: this public surface is identical on every architecture;
//! arch differences live strictly in the leaf kernels below.

pub mod add_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod add_f32_neon;

use crate::cpu::dispatch::Arch;

/// Non-temporal-store crossover for the AVX kernels.
///
/// add is memory-bound and the AVX1/AVX2 kernels use streaming (`vmovntps`)
/// stores, which bypass cache. Measured on the reference i5-3450 (`cargo bench
/// --bench kernels -- add_f32`): below this size the streaming kernel is SLOWER
/// than rustc's auto-vectorised scalar loop (4 K: 0.76×, 64 K: 0.86×), because
/// the result still fits in cache and NT stores just throw away locality. Above
/// it, streaming wins (1 M: 1.37×). So we only leave scalar for large N.
/// (~1 M f32 = 4 MB out, comfortably past this box's per-core L2 and into L3.)
const NON_TEMPORAL_MIN: usize = 1 << 20; // 1_048_576 elements

/// Dispatches FP32 ADD to the best kernel the CPU supports (or the forced arch).
/// NOTE: no dedicated SSE2 kernel exists for add yet — pre-AVX x86 falls to
/// scalar. (sub has an SSE2 kernel; add's is a known gap, not a regression.)
pub fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    // A forced arch is an explicit request (bench/debug) — honour it verbatim,
    // skipping the size heuristic so the forced kernel is what actually runs.
    let forced = crate::cpu::dispatch::forced_arch().is_some();
    let arch = crate::cpu::dispatch::active_arch();

    match arch {
        #[cfg(target_arch = "x86_64")]
        Arch::Avx512 if forced || a.len() >= NON_TEMPORAL_MIN => unsafe { add_f32_avx512::add(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx2 if forced || a.len() >= NON_TEMPORAL_MIN => unsafe { add_f32_avx2::add(a, b, res) },
        #[cfg(target_arch = "x86_64")]
        Arch::Avx1 if forced || a.len() >= NON_TEMPORAL_MIN => unsafe { add_f32_avx1::add(a, b, res) },

        #[cfg(target_arch = "aarch64")]
        Arch::Neon => unsafe { add_f32_neon::add(a, b, res) },

        _ => add_f32_scalar::add(a, b, res),
    }
}
