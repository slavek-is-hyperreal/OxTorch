#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise ReLU for F32 tensors.
pub fn relu_f32(in_buf: &[f32], out_buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { relu_f32_avx(in_buf, out_buf) }; }
    }
    #[cfg(target_arch = "aarch64")] {
        return unsafe { relu_f32_neon(in_buf, out_buf) };
    }
    // GPR fallback
    for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.max(0.0); }
}

/// In-place ReLU for F32 tensors.
pub fn relu_f32_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { relu_f32_avx_inplace(buf) }; }
    }
    #[cfg(target_arch = "aarch64")] {
        return unsafe { relu_f32_neon_inplace(buf) };
    }
    for x in buf.iter_mut() { *x = x.max(0.0); }
}

#[cfg(target_arch = "aarch64")]
unsafe fn relu_f32_neon(in_buf: &[f32], out_buf: &mut [f32]) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_f32(0.0);
    let n4 = (in_buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(in_buf.as_ptr().add(i));
        vst1q_f32(out_buf.as_mut_ptr().add(i), vmaxq_f32(v, zero));
    }
    for i in n4..in_buf.len() { out_buf[i] = in_buf[i].max(0.0); }
}

#[cfg(target_arch = "aarch64")]
unsafe fn relu_f32_neon_inplace(buf: &mut [f32]) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_f32(0.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = vld1q_f32(ptr);
        vst1q_f32(ptr, vmaxq_f32(v, zero));
    }
    for x in buf[n4..].iter_mut() { *x = x.max(0.0); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn relu_f32_avx_inplace(buf: &mut [f32]) {
    let zero = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i);
        let v = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(ptr, _mm256_max_ps(v, zero));
    }
    for x in buf[n8..].iter_mut() { *x = x.max(0.0); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn relu_f32_avx(in_buf: &[f32], out_buf: &mut [f32]) {
    let zero = _mm256_setzero_ps(); let n8 = (in_buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let v = _mm256_loadu_ps(in_buf.as_ptr().add(i));
        _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
    }
    for i in n8..in_buf.len() { out_buf[i] = in_buf[i].max(0.0); }
}
