//! Specialized AVX1 Implementation for Subtraction (Ivy Bridge).
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn sub(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // Ivy Bridge Port 1 Affinity Optimization:
    // Unrolled by 8 to hide the subtraction latency of 3 cycles on a single port.
    while i + 63 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let a4 = _mm256_loadu_ps(a.as_ptr().add(i + 32));
        let a5 = _mm256_loadu_ps(a.as_ptr().add(i + 40));
        let a6 = _mm256_loadu_ps(a.as_ptr().add(i + 48));
        let a7 = _mm256_loadu_ps(a.as_ptr().add(i + 56));

        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        let b4 = _mm256_loadu_ps(b.as_ptr().add(i + 32));
        let b5 = _mm256_loadu_ps(b.as_ptr().add(i + 40));
        let b6 = _mm256_loadu_ps(b.as_ptr().add(i + 48));
        let b7 = _mm256_loadu_ps(b.as_ptr().add(i + 56));

        let r0 = _mm256_sub_ps(a0, b0);
        let r1 = _mm256_sub_ps(a1, b1);
        let r2 = _mm256_sub_ps(a2, b2);
        let r3 = _mm256_sub_ps(a3, b3);
        let r4 = _mm256_sub_ps(a4, b4);
        let r5 = _mm256_sub_ps(a5, b5);
        let r6 = _mm256_sub_ps(a6, b6);
        let r7 = _mm256_sub_ps(a7, b7);

        // --- Scientific-Grade Safety Hook ---
        // Using _mm256_stream_ps for Non-Temporal stores to bypass cache hierarchy during MSTS streaming.
        // Requires 64-byte alignment (guaranteed by TensorPool 2.1).
        _mm256_stream_ps(res.as_mut_ptr().add(i), r0);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 8), r1);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 16), r2);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 24), r3);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 32), r4);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 40), r5);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 48), r6);
        _mm256_stream_ps(res.as_mut_ptr().add(i + 56), r7);

        i += 64;
    }

    // Scalar fallback for remainder
    for rem in i..n {
        res[rem] = a[rem] - b[rem];
    }
}
