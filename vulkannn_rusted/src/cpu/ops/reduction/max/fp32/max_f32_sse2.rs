//! SSE2 f32 max (_mm_max_ps). Ignores NaN (legacy). BENCH: PENDING (unary harness).
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
pub unsafe fn max(buf: &[f32], initial: f32) -> f32 {
    let mut m = _mm_set1_ps(initial); let n4 = (buf.len()/4)*4;
    for i in (0..n4).step_by(4) { m = _mm_max_ps(m, _mm_loadu_ps(buf.as_ptr().add(i))); }
    let mut t = [0f32;4]; _mm_storeu_ps(t.as_mut_ptr(), m);
    let mut r = t.iter().fold(initial, |a,&b| a.max(b));
    for &x in &buf[n4..] { r = r.max(x); } r
}
