#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise ReLU for F16 tensors.
pub fn relu_f16(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { relu_f16_f16c(in_buf, out_buf) };
        }
    }
    #[cfg(target_arch = "aarch64")] {
        return relu_f16_neon(in_buf, out_buf);
    }
    for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32().max(0.0)); }
}

#[cfg(target_arch = "aarch64")]
fn relu_f16_neon(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    use std::arch::aarch64::*;
    // Using f32 vectorization as a reliable fallback for NEON on Rust
    let n4 = (in_buf.len() / 4) * 4;
    let zero = unsafe { vdupq_n_f32(0.0) };
    for i in (0..n4).step_by(4) {
        unsafe {
            let f32_vals = [
                in_buf[i].to_f32(), in_buf[i+1].to_f32(),
                in_buf[i+2].to_f32(), in_buf[i+3].to_f32(),
            ];
            let v = vld1q_f32(f32_vals.as_ptr());
            let res = vmaxq_f32(v, zero);
            let mut out_vals = [0.0f32; 4];
            vst1q_f32(out_vals.as_mut_ptr(), res);
            for k in 0..4 { out_buf[i+k] = half::f16::from_f32(out_vals[k]); }
        }
    }
    for i in n4..in_buf.len() { out_buf[i] = half::f16::from_f32(in_buf[i].to_f32().max(0.0)); }
}

/// In-place ReLU for F16 tensors.
pub fn relu_f16_inplace(buf: &mut [half::f16]) {
    for x in buf.iter_mut() { *x = half::f16::from_f32(x.to_f32().max(0.0)); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn relu_f16_f16c(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    let zero = _mm256_setzero_ps(); let n8 = (in_buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_in = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_cvtph_ps(h_in);
        let f_res = _mm256_max_ps(f_vec, zero);
        _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(f_res));
    }
    for i in n8..in_buf.len() { out_buf[i] = half::f16::from_f32(in_buf[i].to_f32().max(0.0)); }
}
