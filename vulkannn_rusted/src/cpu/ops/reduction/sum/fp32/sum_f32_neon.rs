//! NEON f32 sum — f64 accumulation (widen f32->f64 via vcvt_f64_f32).
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn sum(buf: &[f32]) -> f64 {
    let mut a0 = vdupq_n_f64(0.0);
    let mut a1 = vdupq_n_f64(0.0);
    let n4 = (buf.len() / 4) * 4;
    let mut i = 0;
    while i < n4 {
        let v = vld1q_f32(buf.as_ptr().add(i));
        a0 = vaddq_f64(a0, vcvt_f64_f32(vget_low_f32(v)));
        a1 = vaddq_f64(a1, vcvt_high_f64_f32(v));
        i += 4;
    }
    let s = vaddq_f64(a0, a1);
    let mut acc = vaddvq_f64(s);
    for &x in &buf[n4..] { acc += x as f64; }
    acc
}
