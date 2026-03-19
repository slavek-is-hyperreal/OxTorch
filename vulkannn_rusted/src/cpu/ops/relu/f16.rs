use half;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

pub fn relu_f16(src: &[half::f16], dst: &mut [half::f16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
    {
        return unsafe { relu_f16_f16c(src, dst) };
    }
    for (o, &i) in dst.iter_mut().zip(src.iter()) {
        *o = if i.to_f32() > 0.0 { i } else { half::f16::ZERO };
    }
}

pub fn relu_f16_inplace(buf: &mut [half::f16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
    {
        return unsafe { relu_f16_f16c_inplace(buf) };
    }
    for x in buf.iter_mut() {
        if x.to_f32() < 0.0 { *x = half::f16::ZERO; }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
unsafe fn relu_f16_f16c(src: &[half::f16], dst: &mut [half::f16]) {
    let zero = _mm256_setzero_ps();
    let n8 = (src.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_cvtph_ps(h_vec);
        let res_f = _mm256_max_ps(f_vec, zero);
        let res_h = _mm256_cvtps_ph(res_f, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, res_h);
    }
    for j in n8..src.len() {
        dst[j] = if src[j].to_f32() > 0.0 { src[j] } else { half::f16::ZERO };
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
unsafe fn relu_f16_f16c_inplace(buf: &mut [half::f16]) {
    let zero = _mm256_setzero_ps();
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i) as *mut __m128i;
        let h_vec = _mm_loadu_si128(ptr);
        let f_vec = _mm256_cvtph_ps(h_vec);
        let res_f = _mm256_max_ps(f_vec, zero);
        let res_h = _mm256_cvtps_ph(res_f, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(ptr, res_h);
    }
    for x in buf[n8..].iter_mut() {
        if x.to_f32() < 0.0 { *x = half::f16::ZERO; }
    }
}
