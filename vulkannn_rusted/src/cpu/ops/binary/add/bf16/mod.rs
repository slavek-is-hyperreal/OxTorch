mod add_bf16_generic;
#[cfg(target_arch = "x86_64")]
pub mod add_bf16_avx;

#[cfg(target_arch = "x86_64")]
pub use add_bf16_avx::add_bf16_avx_serial;

/// Precision-specific dispatcher for BF16 addition.
pub fn add_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe {
                add_bf16_avx_serial(a, b, res);
            }
            return;
        }
    }

    add_bf16_generic::add_bf16(a, b, res);
}
