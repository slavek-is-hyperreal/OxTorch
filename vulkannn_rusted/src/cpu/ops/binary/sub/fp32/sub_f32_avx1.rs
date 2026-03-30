#[cfg(target_arch = "x86_64")]
#[allow(unused_assignments)]
pub unsafe fn sub_f32_avx1(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n32 = (n / 32) * 32;
    
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();
    let mut ptr_res = res.as_mut_ptr();
    let mut count = n32 / 32;

    if count > 0 {
        std::arch::asm!(
            "2:",
            // Software Prefetching: 512 bytes ahead (L2 distance)
            "prefetchnta [{ptr_a} + 512]",
            "prefetchnta [{ptr_b} + 512]",
            
            // Load Tensor A (256-bit)
            "vmovups ymm0, [{ptr_a}]",
            "vmovups ymm1, [{ptr_a} + 32]",
            "vmovups ymm2, [{ptr_a} + 64]",
            "vmovups ymm3, [{ptr_a} + 96]",
            
            // Subtraction (Fused with load of Tensor B)
            "vsubps ymm8, ymm0, [{ptr_b}]",
            "vsubps ymm9, ymm1, [{ptr_b} + 32]",
            "vsubps ymm10, ymm2, [{ptr_b} + 64]",
            "vsubps ymm11, ymm3, [{ptr_b} + 96]",
            
            // Non-Temporal Stores (Bypass cache)
            "vmovntps [{ptr_res}], ymm8",
            "vmovntps [{ptr_res} + 32], ymm9",
            "vmovntps [{ptr_res} + 64], ymm10",
            "vmovntps [{ptr_res} + 96], ymm11",
            
            "add {ptr_a}, 128",
            "add {ptr_b}, 128",
            "add {ptr_res}, 128",
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
    
    // Cleanup 
    let processed = n32;
    for i in processed..n {
        res[i] = a[i] - b[i];
    }
}
