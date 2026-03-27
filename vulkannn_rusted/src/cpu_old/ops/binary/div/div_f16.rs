#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise division for F16 tensors with multi-architecture support.
pub fn div_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    const PAR_THRESHOLD: usize = 256_000;
    if a.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            div_f16_serial(ac, bc, rc);
        });
    } else {
        div_f16_serial(a, b, res);
    }
}

fn div_f16_serial(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { div_f16_f16c(a, b, res) };
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return div_f16_neon(a, b, res);
    }

    div_f16_scalar(a, b, res);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn div_f16_f16c(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    let n8 = (a.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let va = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        let vr = _mm256_div_ps(va, vb);
        _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT));
    }
    div_f16_scalar(&a[n8..], &b[n8..], &mut res[n8..]);
}

#[cfg(target_arch = "aarch64")]
fn div_f16_neon(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    use std::arch::aarch64::*;
    let n4 = (a.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        unsafe {
            let va = vcvt_f32_f16(vld1_u16(a.as_ptr().add(i) as *const u16));
            let vb = vcvt_f32_f16(vld1_u16(b.as_ptr().add(i) as *const u16));
            let vr = vdivq_f32(va, vb);
            vst1_u16(res.as_mut_ptr().add(i) as *mut u16, vcvt_f16_f32(vr));
        }
    }
    div_f16_scalar(&a[n4..], &b[n4..], &mut res[n4..]);
}

fn div_f16_scalar(a: &[half::f16], b: &[half::f16], res: &mut [half::f16]) {
    for i in 0..a.len() {
        let val_b = b[i].to_f32();
        res[i] = half::f16::from_f32(if val_b != 0.0 { a[i].to_f32() / val_b } else { 0.0 });
    }
}
