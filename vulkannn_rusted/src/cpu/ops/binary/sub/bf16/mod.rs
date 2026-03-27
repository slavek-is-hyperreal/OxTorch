mod sub_bf16_generic;
#[cfg(target_arch = "x86_64")]
pub mod sub_bf16_avx;

#[cfg(target_arch = "x86_64")]
pub use sub_bf16_avx::sub_bf16_avx_serial;

pub fn sub_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { sub_bf16_avx_serial(a, b, res); }
            return;
        }
    }
    sub_bf16_generic::sub_bf16(a, b, res);
}
