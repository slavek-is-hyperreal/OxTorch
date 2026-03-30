//! Static Dispatcher for the ATAN2 FP32 Specialization Matrix.
//! Utilizes runtime feature detection to select the optimal scientific kernel.

pub mod atan2_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx;
#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod atan2_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod atan2_f32_neon;
#[cfg(target_arch = "aarch64")]
pub mod atan2_f32_sve;
#[cfg(target_arch = "aarch64")]
pub mod atan2_f32_sve2;

/// Dispatches the Atan2 operation to the best available hardware kernel.
pub fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { atan2_f32_avx512::atan2(y, x, res) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { atan2_f32_avx2::atan2(y, x, res) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { atan2_f32_avx1::atan2(y, x, res) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sve2") {
             return unsafe { atan2_f32_sve2::atan2(y, x, res) };
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
             return unsafe { atan2_f32_sve::atan2(y, x, res) };
        }
        return unsafe { atan2_f32_neon::atan2(y, x, res) };
    }

    // Default Fallback
    atan2_f32_scalar::atan2(y, x, res);
}
