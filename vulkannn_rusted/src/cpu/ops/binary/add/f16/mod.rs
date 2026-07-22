//! F16 ADD — Tier II serial dispatcher. F16C is its own x86 tier (as legacy);
//! no avx2/avx512 f16 kernel exists, so none is fabricated (Rule 1).

pub mod add_f16_scalar;

#[cfg(target_arch = "x86_64")]
pub mod add_f16_f16c;

#[cfg(target_arch = "aarch64")]
pub mod add_f16_neon;

pub fn add_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { add_f16_f16c::add_f16_f16c(a, b, res) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { add_f16_neon::add_f16_neon(a, b, res) };
    }
    add_f16_scalar::add(a, b, res);
}
