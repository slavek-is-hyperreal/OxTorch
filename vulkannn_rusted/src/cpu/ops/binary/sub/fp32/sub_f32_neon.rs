#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn sub_f32_neon(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n16 = (n / 16) * 16;
    
    for i in (0..n16).step_by(16) {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let va2 = vld1q_f32(a.as_ptr().add(i + 8));
        let va3 = vld1q_f32(a.as_ptr().add(i + 12));
        
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
        let vb3 = vld1q_f32(b.as_ptr().add(i + 12));
        
        let vr0 = vsubq_f32(va0, vb0);
        let vr1 = vsubq_f32(va1, vb1);
        let vr2 = vsubq_f32(va2, vb2);
        let vr3 = vsubq_f32(va3, vb3);
        
        vst1q_f32(res.as_mut_ptr().add(i), vr0);
        vst1q_f32(res.as_mut_ptr().add(i + 4), vr1);
        vst1q_f32(res.as_mut_ptr().add(i + 8), vr2);
        vst1q_f32(res.as_mut_ptr().add(i + 12), vr3);
    }
    
    for i in n16..n {
        res[i] = a[i] - b[i];
    }
}
