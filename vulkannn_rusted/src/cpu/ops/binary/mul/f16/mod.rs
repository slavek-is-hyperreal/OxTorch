//! F16 MUL — Tier II serial dispatcher.
//!
//! Runtime feature detection. F16C is its own x86 tier (hardware f16<->f32);
//! legacy provides no avx2/avx512 f16 kernel, so we don't fabricate one (Rule 1).
//! Not routed through `active_arch()` because F16C is orthogonal to the SSE/AVX
//! width ladder — it is detected directly, exactly as legacy did.

pub mod mul_f16_scalar;

#[cfg(target_arch = "x86_64")]
pub mod mul_f16_f16c;

#[cfg(target_arch = "aarch64")]
pub mod mul_f16_neon;

pub fn mul_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { mul_f16_f16c::mul_f16_f16c(a, b, res) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { mul_f16_neon::mul_f16_neon(a, b, res) };
    }
    mul_f16_scalar::mul(a, b, res);
}
