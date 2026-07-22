//! I8 ReLU — Tier II serial dispatcher (max(x, 0)).
//! Tiers: scalar / sse4.1 / avx2 / neon. SSE4.1 is the no-AVX2 x86 tier
//! (`_mm_max_epi8`; SSE2 has no signed byte max). Dispatched on direct feature
//! detection because `active_arch` does not model SSE4.1.
//!
//! NOTE: legacy had only avx2 + scalar. A GPR-only SWAR sign-mask tier is a TODO
//! (deliberately NOT hand-rolled here — cf. the carry-leak bug found in the i8
//! add SWAR; §8 also says don't over-invest in memory-bound ops).

pub mod relu_i8_scalar;

#[cfg(target_arch = "x86_64")]
pub mod relu_i8_sse41;
#[cfg(target_arch = "x86_64")]
pub mod relu_i8_avx2;

#[cfg(target_arch = "aarch64")]
pub mod relu_i8_neon;

pub fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
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
    relu_i8_scalar::relu(in_buf, out_buf);
}

pub fn relu_inplace(buf: &mut [i8]) {
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
    relu_i8_scalar::relu_inplace(buf);
}
