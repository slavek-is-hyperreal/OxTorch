#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn sub_f32_sse2(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    let n4 = (n / 4) * 4;
    
    for i in (0..n4).step_by(4) {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vr = _mm_sub_ps(va, vb);
        _mm_storeu_ps(res.as_mut_ptr().add(i), vr);
    }
    
    for i in n4..n {
        res[i] = a[i] - b[i];
    }
}
