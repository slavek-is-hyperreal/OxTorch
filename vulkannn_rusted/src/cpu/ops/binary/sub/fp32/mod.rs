pub mod sub_f32_scalar;
pub mod sub_f32_sse2;
pub mod sub_f32_avx1;
pub mod sub_f32_avx2;

use sub_f32_scalar::sub_f32_scalar;

#[cfg(target_arch = "x86_64")]
use sub_f32_sse2::sub_f32_sse2;
#[cfg(target_arch = "x86_64")]
use sub_f32_avx1::sub_f32_avx1;
#[cfg(target_arch = "x86_64")]
use sub_f32_avx2::sub_f32_avx2;

pub fn sub(a: &[f32], b: &[f32], res: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sub_f32_avx2(a, b, res) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { sub_f32_avx1(a, b, res) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { sub_f32_sse2(a, b, res) };
        }
    }

    sub_f32_scalar(a, b, res);
}
