#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise power for F32 tensors.
/// Specializes for exponent=2.0 (ReLU^2 case).
pub fn pow_f32(in_buf: &[f32], out_buf: &mut [f32], exponent: f32) {
    if exponent == 2.0 {
        #[cfg(target_arch = "x86_64")] {
            if is_x86_feature_detected!("avx") { return unsafe { pow2_f32_avx(in_buf, out_buf) }; }
        }
        #[cfg(target_arch = "aarch64")] {
            return unsafe { pow2_f32_neon(in_buf, out_buf) };
        }
        // GPR fallback
        for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x * x; }
    } else {
        for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.powf(exponent); }
    }
}

/// In-place power for F32 tensors.
pub fn pow_f32_inplace(buf: &mut [f32], exponent: f32) {
    if exponent == 2.0 {
        #[cfg(target_arch = "x86_64")] {
            if is_x86_feature_detected!("avx") { return unsafe { pow2_f32_avx_inplace(buf) }; }
        }
        #[cfg(target_arch = "aarch64")] {
            return unsafe { pow2_f32_neon_inplace(buf) };
        }
        for x in buf.iter_mut() { *x = *x * *x; }
    } else {
        for x in buf.iter_mut() { *x = x.powf(exponent); }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn pow2_f32_neon(in_buf: &[f32], out_buf: &mut [f32]) {
    use std::arch::aarch64::*;
    let n4 = (in_buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(in_buf.as_ptr().add(i));
        vst1q_f32(out_buf.as_mut_ptr().add(i), vmulq_f32(v, v));
    }
    for i in n4..in_buf.len() { out_buf[i] = in_buf[i] * in_buf[i]; }
}

#[cfg(target_arch = "aarch64")]
unsafe fn pow2_f32_neon_inplace(buf: &mut [f32]) {
    use std::arch::aarch64::*;
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = vld1q_f32(ptr);
        vst1q_f32(ptr, vmulq_f32(v, v));
    }
    for x in buf[n4..].iter_mut() { *x = *x * *x; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pow2_f32_avx_inplace(buf: &mut [f32]) {
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(v, v));
    }
    for x in buf[n8..].iter_mut() { *x = *x * *x; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pow2_f32_avx(in_buf: &[f32], out_buf: &mut [f32]) {
    let n8 = (in_buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(in_buf.as_ptr().add(i));
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_mul_ps(v, v));
    }
    for i in n8..in_buf.len() { out_buf[i] = in_buf[i] * in_buf[i]; }
}
