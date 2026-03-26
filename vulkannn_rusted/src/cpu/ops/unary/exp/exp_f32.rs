#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise Exp for F32 tensors with multi-architecture support.
pub fn exp_f32(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { exp_f32_avx_inplace(buf) }; }
        if is_x86_feature_detected!("sse2") { return unsafe { exp_f32_sse2_inplace(buf) }; }
    }
    #[cfg(target_arch = "aarch64")] { return unsafe { exp_f32_neon_inplace(buf) }; }
    
    use rayon::prelude::*;
    buf.par_iter_mut().for_each(|x| *x = x.exp());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn exp_f32_avx_inplace(buf: &mut [f32]) {
    // Logic from reductions.rs:18-40
    let log2_e = _mm256_set1_ps(1.4426950408889634); let ln2_hi = _mm256_set1_ps(0.6931457519);
    let ln2_lo = _mm256_set1_ps(1.4286067653e-6); let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-88.3762626647949); let magic = _mm256_set1_ps(12582912.0);
    let ec1 = _mm256_set1_ps(1.0); let ec2 = _mm256_set1_ps(0.5);
    let ec3 = _mm256_set1_ps(0.16666667163); let ec4 = _mm256_set1_ps(0.04166648536);
    let ec5 = _mm256_set1_ps(0.00833336077); let ec6 = _mm256_set1_ps(0.00138925374);
    let b128 = _mm_set1_epi32(127);
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i); let mut x = _mm256_min_ps(_mm256_loadu_ps(ptr), max_x); x = _mm256_max_ps(x, min_x);
        let mut n_f = _mm256_add_ps(_mm256_mul_ps(x, log2_e), magic); let nb = _mm256_castps_si256(n_f); n_f = _mm256_sub_ps(n_f, magic);
        let f = _mm256_sub_ps(_mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)), _mm256_mul_ps(n_f, ln2_lo));
        let mut p = ec6; p = _mm256_add_ps(_mm256_mul_ps(f, p), ec5); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec4);
        p = _mm256_add_ps(_mm256_mul_ps(f, p), ec3); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec2);
        p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1);
        let pow2n = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<0>(nb), b128), 23)), _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<1>(nb), b128), 23)));
        _mm256_storeu_ps(ptr, _mm256_mul_ps(p, pow2n));
    }
    for x in buf[n8..].iter_mut() { *x = x.exp(); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn exp_f32_sse2_inplace(buf: &mut [f32]) {
    // Logic from reductions.rs:44-52
    let d256 = _mm_set1_ps(1.0/256.0); let one = _mm_set1_ps(1.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i); let mut v = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ptr), d256), one);
        for _ in 0..8 { v = _mm_mul_ps(v, v); } _mm_storeu_ps(ptr, v);
    }
    for x in buf[n4..].iter_mut() { *x = x.exp(); }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp_f32_neon_inplace(buf: &mut [f32]) {
    use std::arch::aarch64::*;
    // Logic from reductions.rs:56-64
    let d256 = vdupq_n_f32(1.0/256.0); let one = vdupq_n_f32(1.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i); let mut v = vfmaq_f32(one, vld1q_f32(ptr), d256);
        for _ in 0..8 { v = vmulq_f32(v, v); } vst1q_f32(ptr, v);
    }
    for x in buf[n4..].iter_mut() { *x = x.exp(); }
}
