//! FP32 Specialization Matrix Dispatcher.
//! Coordinates between Scalar, AVX, AVX2, AVX512, NEON, and SVE kernels.

pub mod add_f32_scalar;

#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx;
#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx1;
#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub mod add_f32_avx512;

#[cfg(target_arch = "aarch64")]
pub mod add_f32_neon;
#[cfg(target_arch = "aarch64")]
pub mod add_f32_sve;
#[cfg(target_arch = "aarch64")]
pub mod add_f32_sve2;

/// Dispatches to the most efficient FP32 ADD kernel available at compile time.
/// This matches the OxTorch Scientific-Grade "Highest Available" policy.
pub fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    // --- x86_64 Dispatch ---
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    { 
        unsafe { add_f32_avx512::add(a, b, res); }
        return;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { 
        unsafe { add_f32_avx2::add(a, b, res); }
        return;
    }

    // Ivy Bridge / standard AVX 
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        // Internal decision: if we don't have AVX2, we use our specialized AVX1 Port-1 kernel.
        unsafe { add_f32_avx1::add(a, b, res); }
        return;
    }

    // --- ARM AArch64 Dispatch ---
    #[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
    { 
        unsafe { add_f32_sve2::add(a, b, res); }
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
    { 
        unsafe { add_f32_sve::add(a, b, res); }
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    { 
        unsafe { add_f32_neon::add(a, b, res); }
        return;
    }

    // --- Fallback ---
    add_f32_scalar::add(a, b, res);
}
