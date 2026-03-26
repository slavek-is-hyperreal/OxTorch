#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Max reduction for F32 tensors.
pub fn max_f32(buf: &[f32], initial: f32) -> f32 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        use rayon::prelude::*;
        return buf.par_chunks(PAR_LIMIT).map(|c| max_f32_serial(c, initial)).reduce(|| initial, |a, b| a.max(b));
    }
    max_f32_serial(buf, initial)
}

fn max_f32_serial(buf: &[f32], initial: f32) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { max_f32_avx(buf, initial) }; }
    }
    #[cfg(target_arch = "aarch64")] {
        return unsafe { max_f32_neon(buf, initial) };
    }
    buf.iter().fold(initial, |a, &b| a.max(b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_f32_avx(buf: &[f32], initial: f32) -> f32 {
    let mut max_v = _mm256_set1_ps(initial); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) { max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(buf.as_ptr().add(i))); }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n8..] { m = m.max(x); } m
}

#[cfg(target_arch = "aarch64")]
unsafe fn max_f32_neon(buf: &[f32], initial: f32) -> f32 {
    use std::arch::aarch64::*;
    let mut max_v = vdupq_n_f32(initial);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        max_v = vmaxq_f32(max_v, vld1q_f32(buf.as_ptr().add(i)));
    }
    let m = vmaxvq_f32(max_v);
    let mut res = initial.max(m);
    for &x in &buf[n4..] { res = res.max(x); }
    res
}
