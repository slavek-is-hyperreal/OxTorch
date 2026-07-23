//! AVX-512 f32 max (_mm512_max_ps). Ignores NaN. BENCH: PENDING (no AVX-512 here).
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn max(buf: &[f32], initial: f32) -> f32 {
    let mut m = _mm512_set1_ps(initial); let n16 = (buf.len()/16)*16;
    for i in (0..n16).step_by(16) { m = _mm512_max_ps(m, _mm512_loadu_ps(buf.as_ptr().add(i))); }
    let hmax = _mm512_reduce_max_ps(m);
    let mut r = initial.max(hmax);
    for &x in &buf[n16..] { r = r.max(x); } r
}
