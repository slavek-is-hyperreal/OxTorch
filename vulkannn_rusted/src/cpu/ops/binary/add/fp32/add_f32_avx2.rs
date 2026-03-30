//! 256-bit AVX2 SIMD Implementation with FMA-as-ADD Port Saturation.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // FMA-as-ADD Optimization for Zen 4 / Skylake-X.
    // By interleaving VADDPS and VFMADD132PS, we saturate all 4 execution ports (FP0/1/2/3).
    let ones = _mm256_set1_ps(1.0);

    while i + 63 < n {
        // Load Block A
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let a4 = _mm256_loadu_ps(a.as_ptr().add(i + 32));
        let a5 = _mm256_loadu_ps(a.as_ptr().add(i + 40));
        let a6 = _mm256_loadu_ps(a.as_ptr().add(i + 48));
        let a7 = _mm256_loadu_ps(a.as_ptr().add(i + 56));

        // Load Block B
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        let b4 = _mm256_loadu_ps(b.as_ptr().add(i + 32));
        let b5 = _mm256_loadu_ps(b.as_ptr().add(i + 40));
        let b6 = _mm256_loadu_ps(b.as_ptr().add(i + 48));
        let b7 = _mm256_loadu_ps(b.as_ptr().add(i + 56));

        // Compute: Interleave ADD and FMA (Port Saturation)
        let r0 = _mm256_add_ps(a0, b0);                // Port FP2/3
        let r1 = _mm256_fmadd_ps(a1, ones, b1);         // Port FP0/1 (Trick: A*1.0 + B)
        let r2 = _mm256_add_ps(a2, b2);                // Port FP2/3
        let r3 = _mm256_fmadd_ps(a3, ones, b3);         // Port FP0/1 (Trick)
        let r4 = _mm256_add_ps(a4, b4);                // Port FP2/3
        let r5 = _mm256_fmadd_ps(a5, ones, b5);         // Port FP0/1 (Trick)
        let r6 = _mm256_add_ps(a6, b6);                // Port FP2/3
        let r7 = _mm256_fmadd_ps(a7, ones, b7);         // Port FP0/1 (Trick)

        // Store: Non-Temporal (Bypass Cache for Result Tensor)
        _mm256_stream_ps(res.as_ptr().add(i) as *mut f32, r0);
        _mm256_stream_ps(res.as_ptr().add(i + 8) as *mut f32, r1);
        _mm256_stream_ps(res.as_ptr().add(i + 16) as *mut f32, r2);
        _mm256_stream_ps(res.as_ptr().add(i + 24) as *mut f32, r3);
        _mm256_stream_ps(res.as_ptr().add(i + 32) as *mut f32, r4);
        _mm256_stream_ps(res.as_ptr().add(i + 40) as *mut f32, r5);
        _mm256_stream_ps(res.as_ptr().add(i + 48) as *mut f32, r6);
        _mm256_stream_ps(res.as_ptr().add(i + 56) as *mut f32, r7);

        i += 64;
    }

    // Scalar remainder
    for remain in i..n {
        res[remain] = a[remain] + b[remain];
    }
}
