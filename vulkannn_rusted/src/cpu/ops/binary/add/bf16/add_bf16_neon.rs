#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn add_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16]) {
    let n4 = (a.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ba = vld1_u16(a.as_ptr().add(i) as *const u16);
        let bb = vld1_u16(b.as_ptr().add(i) as *const u16);
        
        let va = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(ba), 16));
        let vb = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bb), 16));
        
        let vr = vaddq_f32(va, vb);
        
        let vr_u32 = vshrq_n_u32(vreinterpretq_u32_f32(vr), 16);
        vst1_u16(res.as_mut_ptr().add(i) as *mut u16, vmovn_u32(vr_u32));
    }
    // Tail obsługiwany w mod.rs
}
