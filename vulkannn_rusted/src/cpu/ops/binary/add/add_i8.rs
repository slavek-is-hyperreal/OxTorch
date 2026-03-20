#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise addition for I8 tensors with multi-architecture support (Saturating).
pub fn add_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    const PAR_THRESHOLD: usize = 4_000_000;
    if a.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            add_i8_serial(ac, bc, rc);
        });
    } else {
        add_i8_serial(a, b, res);
    }
}

fn add_i8_serial(a: &[i8], b: &[i8], res: &mut [i8]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            return unsafe { add_i8_avx2(a, b, res) };
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return add_i8_neon(a, b, res);
    }

    // SWAR fallback for CPUs with 64-bit registers but no modern SIMD
    if a.len() >= 8 {
        return add_i8_swar(a, b, res);
    }

    add_i8_scalar(a, b, res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_i8_avx2(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n32 = (a.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, _mm256_adds_epi8(va, vb));
    }
    add_i8_scalar(&a[n32..], &b[n32..], &mut res[n32..]);
}

#[cfg(target_arch = "aarch64")]
fn add_i8_neon(a: &[i8], b: &[i8], res: &mut [i8]) {
    use std::arch::aarch64::*;
    let n16 = (a.len() / 16) * 16;
    for i in (0..n16).step_by(16) {
        unsafe {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            vst1q_s8(res.as_mut_ptr().add(i), vqaddq_s8(va, vb));
        }
    }
    add_i8_scalar(&a[n16..], &b[n16..], &mut res[n16..]);
}

/// GPR-based SWAR addition (8 bytes at a time)
fn add_i8_swar(a: &[i8], b: &[i8], res: &mut [i8]) {
    let n8 = (a.len() / 8) * 8;
    let mask_low7 = 0x7F7F7F7F7F7F7F7F_u64;
    let mask_msb  = 0x8080808080808080_u64;

    for i in (0..n8).step_by(8) {
        unsafe {
            let x = *(a.as_ptr().add(i) as *const u64);
            let y = *(b.as_ptr().add(i) as *const u64);
            // Non-saturating SWAR add (wrapping)
            let sum = ((x & mask_low7).wrapping_add(y & mask_low7)) ^ ((x ^ y) & mask_msb);
            *(res.as_mut_ptr().add(i) as *mut u64) = sum;
        }
    }
    add_i8_scalar(&a[n8..], &b[n8..], &mut res[n8..]);
}

fn add_i8_scalar(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        res[i] = a[i].saturating_add(b[i]);
    }
}
