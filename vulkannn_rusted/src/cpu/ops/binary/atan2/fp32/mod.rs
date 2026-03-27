//! Static Dispatcher for the ATAN2 FP32 Specialization Matrix.
//! Utilizes runtime feature detection to select the optimal scientific kernel.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx;
#[cfg(target_arch = "x86_64")]
pub mod avx1;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub mod sve;
#[cfg(target_arch = "aarch64")]
pub mod sve2;

/// Dispatches the Atan2 operation to the best available hardware kernel.
pub fn atan2(y: &[f32], x: &[f32], res: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::atan2(y, x, res) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::atan2(y, x, res) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { avx1::atan2(y, x, res) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sve2") {
             return unsafe { sve2::atan2(y, x, res) };
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
             return unsafe { sve::atan2(y, x, res) };
        }
        return unsafe { neon::atan2(y, x, res) };
    }

    // Default Fallback
    scalar::atan2(y, x, res);
}
