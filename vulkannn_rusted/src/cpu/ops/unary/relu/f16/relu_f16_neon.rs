//! NEON ReLU for F16 (f32 upcast path). Uses the clean vcvt+vmax route (legacy
//! built the f32 vector element-by-element; result is identical: max(x, 0)).
//!
//! BENCH: PENDING (hw: aarch64/NEON). Reference box is x86 — measure on ARM.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn relu(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    let zero = vdupq_n_f32(0.0);
    let n = in_buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let v = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(in_buf.as_ptr().add(i) as *const u16)));
        let r = vmaxq_f32(v, zero);
        vst1_u16(out_buf.as_mut_ptr().add(i) as *mut u16, vreinterpret_u16_f16(vcvt_f16_f32(r)));
    }
    for i in n4..n {
        out_buf[i] = half::f16::from_f32(in_buf[i].to_f32().max(0.0));
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn relu_inplace(buf: &mut [half::f16]) {
    let zero = vdupq_n_f32(0.0);
    let n = buf.len();
    let n4 = (n / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i) as *mut u16;
        let v = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr)));
        vst1_u16(ptr, vreinterpret_u16_f16(vcvt_f16_f32(vmaxq_f32(v, zero))));
    }
    for x in buf[n4..].iter_mut() {
        *x = half::f16::from_f32(x.to_f32().max(0.0));
    }
}
