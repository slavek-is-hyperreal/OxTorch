//! 512-bit AVX-512 SIMD Implementation with Non-Temporal Stores and Port 0/5 Saturation.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,fma")]
pub unsafe fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let mut i = 0;

    // FMA-as-ADD Optimization for Skylake-X / Zen 4.
    // 512-bit ZMM registers use Ports 0 and 5 for execution.
    let ones = _mm512_set1_ps(1.0);

    // Unroll by 8 (8 * 16 = 128 floats per iteration)
    while i + 127 < n {
        // Prefetch NTA (Non-Temporal) for inputs (looking ahead 2 tiles)
        _mm_prefetch(a.as_ptr().add(i + 128) as *const i8, _MM_HINT_NTA);
        _mm_prefetch(b.as_ptr().add(i + 128) as *const i8, _MM_HINT_NTA);

        // Load Block A
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
        let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
        let a4 = _mm512_loadu_ps(a.as_ptr().add(i + 64));
        let a5 = _mm512_loadu_ps(a.as_ptr().add(i + 80));
        let a6 = _mm512_loadu_ps(a.as_ptr().add(i + 96));
        let a7 = _mm512_loadu_ps(a.as_ptr().add(i + 112));

        // Load Block B
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));
        let b4 = _mm512_loadu_ps(b.as_ptr().add(i + 64));
        let b5 = _mm512_loadu_ps(b.as_ptr().add(i + 80));
        let b6 = _mm512_loadu_ps(b.as_ptr().add(i + 96));
        let b7 = _mm512_loadu_ps(b.as_ptr().add(i + 112));

        // Compute: Interleave ADD and FMA (Port Saturation)
        let r0 = _mm512_add_ps(a0, b0);                // Port 0/5
        let r1 = _mm512_fmadd_ps(a1, ones, b1);         // Port 0/5
        let r2 = _mm512_add_ps(a2, b2);
        let r3 = _mm512_fmadd_ps(a3, ones, b3);
        let r4 = _mm512_add_ps(a4, b4);
        let r5 = _mm512_fmadd_ps(a5, ones, b5);
        let r6 = _mm512_add_ps(a6, b6);
        let r7 = _mm512_fmadd_ps(a7, ones, b7);

        // Store: Non-Temporal (Bypass Cache for Result Tensor)
        _mm512_stream_ps(res.as_ptr().add(i) as *mut f32, r0);
        _mm512_stream_ps(res.as_ptr().add(i + 16) as *mut f32, r1);
        _mm512_stream_ps(res.as_ptr().add(i + 32) as *mut f32, r2);
        _mm512_stream_ps(res.as_ptr().add(i + 48) as *mut f32, r3);
        _mm512_stream_ps(res.as_ptr().add(i + 64) as *mut f32, r4);
        _mm512_stream_ps(res.as_ptr().add(i + 80) as *mut f32, r5);
        _mm512_stream_ps(res.as_ptr().add(i + 96) as *mut f32, r6);
        _mm512_stream_ps(res.as_ptr().add(i + 112) as *mut f32, r7);

        i += 128;
    }

    // Scalar remainder
    for remain in i..n {
        res[remain] = a[remain] + b[remain];
    }
}
