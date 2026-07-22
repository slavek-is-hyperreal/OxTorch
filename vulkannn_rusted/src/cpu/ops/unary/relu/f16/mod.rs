//! F16 ReLU — Tier II serial dispatcher. F16C tier on x86, neon on ARM.

pub mod relu_f16_scalar;

#[cfg(target_arch = "x86_64")]
pub mod relu_f16_f16c;

#[cfg(target_arch = "aarch64")]
pub mod relu_f16_neon;

pub fn relu(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { relu_f16_f16c::relu(in_buf, out_buf) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { relu_f16_neon::relu(in_buf, out_buf) };
    }
    relu_f16_scalar::relu(in_buf, out_buf);
}

pub fn relu_inplace(buf: &mut [half::f16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { relu_f16_f16c::relu_inplace(buf) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { relu_f16_neon::relu_inplace(buf) };
    }
    relu_f16_scalar::relu_inplace(buf);
}
