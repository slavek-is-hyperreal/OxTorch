//! NEON exp for FP32 — same Cephes math as the validated x86 kernels, ARMv8.
//! Uses vcvtnq (round-to-nearest) for n and vbslq for edge selects. Coeffs:
//! docs/kernel_specs/exp_spec.md.
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn exp4(x: float32x4_t) -> float32x4_t {
    let log2ef = vdupq_n_f32(1.44269504088896341);
    let c1 = vdupq_n_f32(0.693359375);
    let c2 = vdupq_n_f32(-2.12194440e-4);
    let maxlogf = vdupq_n_f32(88.72283905206835);
    let minlogf = vdupq_n_f32(-103.278929903431851103);

    let xc = vminq_f32(vmaxq_f32(x, minlogf), maxlogf);
    let n = vcvtnq_s32_f32(vmulq_f32(log2ef, xc)); // round-to-nearest
    let fn_ = vcvtq_f32_s32(n);

    let mut g = vsubq_f32(xc, vmulq_f32(fn_, c1));
    g = vsubq_f32(g, vmulq_f32(fn_, c2));

    let mut p = vdupq_n_f32(1.9875691500E-4);
    p = vaddq_f32(vmulq_f32(p, g), vdupq_n_f32(1.3981999507E-3));
    p = vaddq_f32(vmulq_f32(p, g), vdupq_n_f32(8.3334519073E-3));
    p = vaddq_f32(vmulq_f32(p, g), vdupq_n_f32(4.1665795894E-2));
    p = vaddq_f32(vmulq_f32(p, g), vdupq_n_f32(1.6666665459E-1));
    p = vaddq_f32(vmulq_f32(p, g), vdupq_n_f32(5.0000001201E-1));

    let gg = vmulq_f32(g, g);
    let mut eg = vaddq_f32(vmulq_f32(p, gg), g);
    eg = vaddq_f32(eg, vdupq_n_f32(1.0));

    // two-step ldexp (denormal-safe)
    let bias = vdupq_n_s32(127);
    let n1 = vshrq_n_s32::<1>(n);
    let n2 = vsubq_s32(n, n1);
    let pow2a = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, bias)));
    let pow2b = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n2, bias)));
    let mut res = vmulq_f32(vmulq_f32(eg, pow2a), pow2b);

    // Edge selects via vbslq (mask picks first arg).
    let inf = vdupq_n_f32(f32::INFINITY);
    let ov = vcgtq_f32(x, maxlogf);                 // x > MAXLOGF (also +inf)
    let un = vcltq_f32(x, minlogf);                 // x < MINLOGF (also -inf)
    let nan = vmvnq_u32(vceqq_f32(x, x));           // x != x  -> NaN
    res = vbslq_f32(ov, inf, res);
    res = vbslq_f32(un, vdupq_n_f32(0.0), res);
    res = vbslq_f32(nan, x, res);
    res
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn exp(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        vst1q_f32(out_buf.as_mut_ptr().add(i), exp4(vld1q_f32(in_buf.as_ptr().add(i))));
    }
    for i in n4..n {
        out_buf[i] = in_buf[i].exp();
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn exp_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i);
        vst1q_f32(ptr, exp4(vld1q_f32(ptr)));
    }
    for x in buf[n4..].iter_mut() {
        *x = x.exp();
    }
}
