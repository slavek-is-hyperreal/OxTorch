//! I8 MUL — Tier II serial dispatcher (saturating).
//!
//! Runtime feature detection. Legacy provides only avx2 (x86) and neon; no
//! avx1/sse2 i8 kernel, so we don't fabricate one (Rule 1; §8 memory-bound).

pub mod mul_i8_scalar;

#[cfg(target_arch = "x86_64")]
pub mod mul_i8_avx2;

#[cfg(target_arch = "aarch64")]
pub mod mul_i8_neon;

pub fn mul_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { mul_i8_avx2::mul_i8_avx2(a, b, res) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { mul_i8_neon::mul_i8_neon(a, b, res) };
    }
    mul_i8_scalar::mul(a, b, res);
}
