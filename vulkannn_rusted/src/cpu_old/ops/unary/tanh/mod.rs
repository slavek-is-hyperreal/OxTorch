/// Elementwise Tanh activation for F32 tensors.
pub fn tanh_f32(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        for chunk in buf.chunks_exact_mut(8) {
            unsafe {
                let v = std::arch::x86_64::_mm256_loadu_ps(chunk.as_ptr());
                let res = crate::cpu_old::ops::math_simd::tanh_ps_avx2(v);
                std::arch::x86_64::_mm256_storeu_ps(chunk.as_mut_ptr(), res);
            }
        }
        for x in buf.chunks_exact_mut(8).into_remainder() {
            *x = x.tanh();
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        for chunk in buf.chunks_exact_mut(4) {
            unsafe {
                let v = std::arch::aarch64::vld1q_f32(chunk.as_ptr());
                let res = crate::cpu_old::ops::math_simd::tanh_ps_neon(v);
                std::arch::aarch64::vst1q_f32(chunk.as_mut_ptr(), res);
            }
        }
        for x in buf.chunks_exact_mut(4).into_remainder() {
            *x = x.tanh();
        }
        return;
    }

    for x in buf.iter_mut() {
        *x = x.tanh();
    }
}

pub fn tanh_f16(buf: &mut [half::f16]) {
    let mut f32_buf = crate::tensor::pool::TensorPool::get_f32_buffer(buf.len());
    for i in 0..buf.len() { f32_buf[i] = buf[i].to_f32(); }
    tanh_f32(&mut f32_buf);
    for i in 0..buf.len() { buf[i] = half::f16::from_f32(f32_buf[i]); }
}

pub fn tanh_bf16(buf: &mut [half::bf16]) {
    let mut f32_buf = crate::tensor::pool::TensorPool::get_f32_buffer(buf.len());
    for i in 0..buf.len() { f32_buf[i] = buf[i].to_f32(); }
    tanh_f32(&mut f32_buf);
    for i in 0..buf.len() { buf[i] = half::bf16::from_f32(f32_buf[i]); }
}

pub fn tanh_i8(buf: &mut [i8]) {
    static mut TANH_LUT: [i8; 256] = [0; 256];
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        for i in 0..256 {
            let x = (i as i32 - 128) as f32 / 16.0;
            let res = x.tanh();
            unsafe { TANH_LUT[i] = (res * 127.0).round() as i8; }
        }
    });
    for x in buf.iter_mut() {
        *x = unsafe { TANH_LUT[(*x as i32 + 128) as usize] };
    }
}
