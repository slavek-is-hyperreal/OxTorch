mod mul_bf16_generic;
#[cfg(target_arch = "x86_64")]
pub mod mul_bf16_avx;

pub fn mul_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { mul_bf16_avx::mul_bf16_avx_serial(a, b, res); }
            return;
        }
    }
    mul_bf16_generic::mul_bf16(a, b, res);
}
