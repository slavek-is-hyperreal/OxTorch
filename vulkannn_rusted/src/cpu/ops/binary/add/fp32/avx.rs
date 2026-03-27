//! 256-bit AVX SIMD Implementation for FP32 Addition.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // Unroll by 4 (4 * 8 = 32 floats per iteration)
    while i + 31 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));

        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

        let r0 = _mm256_add_ps(a0, b0);
        let r1 = _mm256_add_ps(a1, b1);
        let r2 = _mm256_add_ps(a2, b2);
        let r3 = _mm256_add_ps(a3, b3);

        _mm256_storeu_ps(res.as_ptr().add(i) as *mut f32, r0);
        _mm256_storeu_ps(res.as_ptr().add(i + 8) as *mut f32, r1);
        _mm256_storeu_ps(res.as_ptr().add(i + 16) as *mut f32, r2);
        _mm256_storeu_ps(res.as_ptr().add(i + 24) as *mut f32, r3);

        i += 32;
    }

    // SIMD remainder (8 floats per iteration)
    while i + 7 < n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(res.as_ptr().add(i) as *mut f32, vr);
        i += 8;
    }

    // Scalar remainder
    for remain in i..n {
        res[remain] = a[remain] + b[remain];
    }
}
