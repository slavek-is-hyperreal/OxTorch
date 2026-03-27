#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise addition for F32 tensors with multi-architecture support.
pub fn add_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    const PAR_THRESHOLD: usize = 4_000_000;
    if a.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            add_f32_serial(ac, bc, rc);
        });
    } else {
        add_f32_serial(a, b, res);
    }
}

fn add_f32_serial(a: &[f32], b: &[f32], res: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            return unsafe { add_f32_avx2(a, b, res) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { add_f32_avx(a, b, res) };
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return add_f32_neon(a, b, res);
    }

    add_f32_scalar(a, b, res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n8 = (a.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(res.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
    }
    add_f32_scalar(&a[n8..], &b[n8..], &mut res[n8..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_f32_avx(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n8 = (a.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(res.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
    }
    add_f32_scalar(&a[n8..], &b[n8..], &mut res[n8..]);
}

#[cfg(target_arch = "aarch64")]
fn add_f32_neon(a: &[f32], b: &[f32], res: &mut [f32]) {
    use std::arch::aarch64::*;
    let n4 = (a.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            vst1q_f32(res.as_mut_ptr().add(i), vaddq_f32(va, vb));
        }
    }
    add_f32_scalar(&a[n4..], &b[n4..], &mut res[n4..]);
}

fn add_f32_scalar(a: &[f32], b: &[f32], res: &mut [f32]) {
    for i in 0..a.len() {
        res[i] = a[i] + b[i];
    }
}
