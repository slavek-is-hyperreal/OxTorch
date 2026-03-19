#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn relu_f32(src: &[f32], dst: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n8 = (src.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i), 
            _mm256_max_ps(_mm256_loadu_ps(src.as_ptr().add(i)), zero)
        );
    }
    // Tail
    for j in n8..src.len() {
        dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn relu_f32_inplace(buf: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let p = buf.as_mut_ptr().add(i);
        _mm256_storeu_ps(p, _mm256_max_ps(_mm256_loadu_ps(p), zero));
    }
    for x in buf[n8..].iter_mut() {
        if *x < 0.0 { *x = 0.0; }
    }
}

// TODO: Add AVX-512 variants if needed for future servers.
