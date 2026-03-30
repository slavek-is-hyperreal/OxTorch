#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_f32_avx2(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n64 = (n / 64) * 64;
    
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();
    let mut ptr_res = res.as_mut_ptr();
    let mut count = n64 / 64;

    if count > 0 {
        std::arch::asm!(
            "2:",
            // Software Prefetching: 512 bytes ahead
            "prefetchnta [{ptr_a} + 512]",
            "prefetchnta [{ptr_b} + 512]",
            
            // Phase 1: Load A
            "vmovups ymm0, [{ptr_a}]",
            "vmovups ymm1, [{ptr_a} + 32]",
            "vmovups ymm2, [{ptr_a} + 64]",
            "vmovups ymm3, [{ptr_a} + 96]",
            "vmovups ymm4, [{ptr_a} + 128]",
            "vmovups ymm5, [{ptr_a} + 160]",
            "vmovups ymm6, [{ptr_a} + 192]",
            "vmovups ymm7, [{ptr_a} + 224]",
            
            // Phase 2: Fused Load-Sub from B
            "vsubps ymm8, ymm0, [{ptr_b}]",
            "vsubps ymm9, ymm1, [{ptr_b} + 32]",
            "vsubps ymm10, ymm2, [{ptr_b} + 64]",
            "vsubps ymm11, ymm3, [{ptr_b} + 96]",
            "vsubps ymm12, ymm4, [{ptr_b} + 128]",
            "vsubps ymm13, ymm5, [{ptr_b} + 160]",
            "vsubps ymm14, ymm6, [{ptr_b} + 192]",
            "vsubps ymm15, ymm7, [{ptr_b} + 224]",
            
            // Phase 3: Non-Temporal Stores
            "vmovntps [{ptr_res}], ymm8",
            "vmovntps [{ptr_res} + 32], ymm9",
            "vmovntps [{ptr_res} + 64], ymm10",
            "vmovntps [{ptr_res} + 96], ymm11",
            "vmovntps [{ptr_res} + 128], ymm12",
            "vmovntps [{ptr_res} + 160], ymm13",
            "vmovntps [{ptr_res} + 192], ymm14",
            "vmovntps [{ptr_res} + 224], ymm15",
            
            "add {ptr_a}, 256",
            "add {ptr_b}, 256",
            "add {ptr_res}, 256",
            "dec {count}",
            "jnz 2b",
            "sfence",
            ptr_a = inout(reg) ptr_a,
            ptr_b = inout(reg) ptr_b,
            ptr_res = inout(reg) ptr_res,
            count = inout(reg) count,
            options(nostack, preserves_flags)
        );
    }
    
    // Remaining elements
    let processed = n64;
    for i in processed..n {
        res[i] = a[i] - b[i];
    }
}
