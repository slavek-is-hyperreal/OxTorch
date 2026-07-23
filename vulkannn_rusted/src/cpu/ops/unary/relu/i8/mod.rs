//! I8 ReLU — Tier II serial dispatcher (max(x, 0)).
//! Tiers: scalar / sse4.1 / avx2 / neon. SSE4.1 is the no-AVX2 x86 tier
//! (`_mm_max_epi8`; SSE2 has no signed byte max). Dispatched on direct feature
//! detection because `active_arch` does not model SSE4.1.
//!
//! NOTE: legacy had only avx2 + scalar. The GPR-only SWAR tier (`relu_i8_swar`)
//! is now filled from the verified catalog (sign-spread mask + AND, exhaustively
//! tested in cpu::swar) — the TODO from dfae1ee is closed. On x86 it is never
//! selected (SSE4.1 wins); it serves targets with no vector unit, and
//! force_arch("swar").

pub mod relu_i8_scalar;
pub mod relu_i8_swar;

#[cfg(target_arch = "x86_64")]
pub mod relu_i8_sse41;
#[cfg(target_arch = "x86_64")]
pub mod relu_i8_avx2;

#[cfg(target_arch = "aarch64")]
pub mod relu_i8_neon;

pub fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
    // Explicit SWAR override (force_arch): honour it verbatim for debugging.
    if crate::cpu::dispatch::forced_arch() == Some(crate::cpu::dispatch::Arch::Swar) {
        return relu_i8_swar::relu(in_buf, out_buf);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { relu_i8_avx2::relu(in_buf, out_buf) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { relu_i8_sse41::relu(in_buf, out_buf) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { relu_i8_neon::relu(in_buf, out_buf) };
    }
    // GPR fallback for targets with no vector unit.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        if in_buf.len() >= 8 {
            return relu_i8_swar::relu(in_buf, out_buf);
        }
    }
    relu_i8_scalar::relu(in_buf, out_buf);
}

pub fn relu_inplace(buf: &mut [i8]) {
    if crate::cpu::dispatch::forced_arch() == Some(crate::cpu::dispatch::Arch::Swar) {
        return relu_i8_swar::relu_inplace(buf);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { relu_i8_avx2::relu_inplace(buf) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { relu_i8_sse41::relu_inplace(buf) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { relu_i8_neon::relu_inplace(buf) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        if buf.len() >= 8 {
            return relu_i8_swar::relu_inplace(buf);
        }
    }
    relu_i8_scalar::relu_inplace(buf);
}
