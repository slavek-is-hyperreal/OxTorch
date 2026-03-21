#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise subtraction for I8 tensors with multi-architecture support (Saturating).
pub fn sub_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    const PAR_THRESHOLD: usize = 4_000_000;
    if a.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            sub_i8_serial(ac, bc, rc);
        });
    } else {
        sub_i8_serial(a, b, res);
    }
}

fn sub_i8_serial(a: &[i8], b: &[i8], res: &mut [i8]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sub_i8_avx2(a, b, res) };
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return sub_i8_neon(a, b, res);
    }

    // SWAR fallback for CPUs with 64-bit registers but no modern SIMD
    if a.len() >= 8 {
        return sub_i8_swar(a, b, res);
    }

    sub_i8_scalar(a, b, res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_i8_avx2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n32 = (a.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, _mm256_subs_epi8(va, vb));
    }
    sub_i8_scalar(&a[n32..], &b[n32..], &mut res[n32..]);
}

#[cfg(target_arch = "aarch64")]
fn sub_i8_neon(a: &[i8], b: &[i8], res: &mut [i8]) {
    use std::arch::aarch64::*;
    let n16 = (a.len() / 16) * 16;
    for i in (0..n16).step_by(16) {
        unsafe {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            vst1q_s8(res.as_mut_ptr().add(i), vqsubq_s8(va, vb));
        }
    }
    sub_i8_scalar(&a[n16..], &b[n16..], &mut res[n16..]);
}

/// GPR-based SWAR subtraction (8 bytes at a time)
fn sub_i8_swar(a: &[i8], b: &[i8], res: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        // On x86_64 we always have at least SSE2; _mm_subs_epi8 handles saturating sub.
        return sub_i8_sse2(a, b, res);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // For other arches without SIMD, keep scalar for now.
        sub_i8_scalar(a, b, res);
    }
}

#[cfg(target_arch = "x86_64")]
fn sub_i8_sse2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n16 = (a.len() / 16) * 16;
    for i in (0..n16).step_by(16) {
        unsafe {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, _mm_subs_epi8(va, vb));
        }
    }
    sub_i8_scalar(&a[n16..], &b[n16..], &mut res[n16..]);
}

fn sub_i8_scalar(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = a[i].saturating_sub(b[i]);
    }
}
