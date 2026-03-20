#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise ReLU for I8 tensors.
pub fn relu_i8(in_buf: &[i8], out_buf: &mut [i8]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") { return unsafe { relu_i8_avx2(in_buf, out_buf) }; }
    }
    for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.max(0); }
}

/// In-place ReLU for I8 tensors.
pub fn relu_i8_inplace(buf: &mut [i8]) {
    for x in buf.iter_mut() { *x = x.max(0); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_i8_avx2(in_buf: &[i8], out_buf: &mut [i8]) {
    let zero = _mm256_setzero_si256(); let n32 = (in_buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let v = _mm256_loadu_si256(in_buf.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(out_buf.as_mut_ptr().add(i) as *mut __m256i, _mm256_max_epi8(v, zero));
    }
    for i in n32..in_buf.len() { out_buf[i] = in_buf[i].max(0); }
}
