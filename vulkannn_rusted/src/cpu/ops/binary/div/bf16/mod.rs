mod div_bf16_generic;
#[cfg(target_arch = "x86_64")]
pub mod div_bf16_avx;

#[cfg(target_arch = "x86_64")]
pub use div_bf16_avx::div_bf16_avx_serial;

pub fn div_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { div_bf16_avx::div_bf16_avx_serial(a, b, res); }
            return;
        }
    }
    div_bf16_generic::div_bf16(a, b, res);
}
