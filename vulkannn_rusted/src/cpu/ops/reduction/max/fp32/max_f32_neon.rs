//! NEON f32 max (vmaxq_f32). Ignores NaN. BENCH: PENDING (aarch64).
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
pub unsafe fn max(buf: &[f32], initial: f32) -> f32 {
    let mut m = vdupq_n_f32(initial); let n4 = (buf.len()/4)*4;
    for i in (0..n4).step_by(4) { m = vmaxq_f32(m, vld1q_f32(buf.as_ptr().add(i))); }
    let mut r = initial.max(vmaxvq_f32(m));
    for &x in &buf[n4..] { r = r.max(x); } r
}
