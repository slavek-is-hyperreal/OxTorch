#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn relu_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx512f") { unsafe { relu_f32_avx512(src, dst); } return; }
        if is_x86_feature_detected!("avx") { unsafe { relu_f32_avx(src, dst); } return; }
        if is_x86_feature_detected!("sse2") { unsafe { relu_f32_sse2(src, dst); } return; }
    }
    #[cfg(target_arch = "aarch64")] { unsafe { relu_f32_neon(src, dst); } return; }
    for (o, &i) in dst.iter_mut().zip(src.iter()) { *o = if i > 0.0 { i } else { 0.0 }; }
}

pub fn relu_f32_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let zero = _mm512_setzero_ps(); let n16 = (buf.len() / 16) * 16;
                for i in (0..n16).step_by(16) { let p = buf.as_mut_ptr().add(i); _mm512_storeu_ps(p, _mm512_max_ps(_mm512_loadu_ps(p), zero)); }
                for x in buf[n16..].iter_mut() { if *x < 0.0 { *x = 0.0; } }
            }
            return;
        }
        if is_x86_feature_detected!("avx") {
            unsafe {
                let zero = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
                for i in (0..n8).step_by(8) { let p = buf.as_mut_ptr().add(i); _mm256_storeu_ps(p, _mm256_max_ps(_mm256_loadu_ps(p), zero)); }
                for x in buf[n8..].iter_mut() { if *x < 0.0 { *x = 0.0; } }
            }
            return;
        }
    }
    for x in buf.iter_mut() { if *x < 0.0 { *x = 0.0; } }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_f32_avx512(src: &[f32], dst: &mut [f32]) {
    let zero = _mm512_setzero_ps(); let n16 = (src.len() / 16) * 16;
    for i in (0..n16).step_by(16) { _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_max_ps(_mm512_loadu_ps(src.as_ptr().add(i)), zero)); }
    for j in n16..src.len() { dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 }; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn relu_f32_avx(src: &[f32], dst: &mut [f32]) {
    let zero = _mm256_setzero_ps(); let n8 = (src.len() / 8) * 8;
    for i in (0..n8).step_by(8) { _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_max_ps(_mm256_loadu_ps(src.as_ptr().add(i)), zero)); }
    for j in n8..src.len() { dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 }; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn relu_f32_sse2(src: &[f32], dst: &mut [f32]) {
    let zero = _mm_setzero_ps(); let n4 = (src.len() / 4) * 4;
    for i in (0..n4).step_by(4) { _mm_storeu_ps(dst.as_mut_ptr().add(i), _mm_max_ps(_mm_loadu_ps(src.as_ptr().add(i)), zero)); }
    for j in n4..src.len() { dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 }; }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn relu_f32_neon(src: &[f32], dst: &mut [f32]) {
    let zero = vdupq_n_f32(0.0); let n4 = (src.len() / 4) * 4;
    for i in (0..n4).step_by(4) { vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(vld1q_f32(src.as_ptr().add(i)), zero)); }
    for j in n4..src.len() { dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 }; }
}

pub fn gelu_f32_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { unsafe { gelu_f32_avx_inplace(buf); } return; }
        if is_x86_feature_detected!("sse2") { unsafe { gelu_f32_sse2_inplace(buf); } return; }
    }
    #[cfg(target_arch = "aarch64")] { unsafe { gelu_f32_neon_inplace(buf); } return; }
    gelu_f32_scalar(buf);
}

pub fn gelu_f32_scalar(buf: &mut [f32]) {
    const K: f32 = 0.7978845608; const C: f32 = 0.044715;
    for x in buf.iter_mut() {
        let v = *x; let inner = K * (v + C * v * v * v); let y = inner.clamp(-9.0, 9.0);
        let e2y = (2.0 * y).exp(); let tanh_v = (e2y - 1.0) / (e2y + 1.0); *x = 0.5 * v * (1.0 + tanh_v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gelu_f32_avx_inplace(buf: &mut [f32]) {
    let vk = _mm256_set1_ps(0.7978845608); let vc = _mm256_set1_ps(0.044715);
    let vhalf = _mm256_set1_ps(0.5); let vone = _mm256_set1_ps(1.0);
    let vtwo = _mm256_set1_ps(2.0); let vclip = _mm256_set1_ps(9.0);
    let vnclip = _mm256_set1_ps(-9.0); let log2e = _mm256_set1_ps(1.4426950408889634);
    let ln2_h = _mm256_set1_ps(0.6931457519); let ln2_l = _mm256_set1_ps(1.4286067653e-6);
    let magic = _mm256_set1_ps(12582912.0); let emax = _mm256_set1_ps(88.376);
    let emin = _mm256_set1_ps(-88.376); let ec1 = _mm256_set1_ps(1.0);
    let ec2 = _mm256_set1_ps(0.5); let ec3 = _mm256_set1_ps(0.16666667163);
    let ec4 = _mm256_set1_ps(0.04166648536); let ec5 = _mm256_set1_ps(0.00833336077);
    let ec6 = _mm256_set1_ps(0.00138925374); let b128 = _mm_set1_epi32(127);

    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i); let x = _mm256_loadu_ps(ptr);
        let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        let inner = _mm256_mul_ps(vk, _mm256_add_ps(x, _mm256_mul_ps(vc, x3)));
        let ic = _mm256_min_ps(_mm256_max_ps(inner, vnclip), vclip);
        let mut xc = _mm256_min_ps(_mm256_mul_ps(vtwo, ic), emax); xc = _mm256_max_ps(xc, emin);
        let nx = _mm256_mul_ps(xc, log2e); let mut n = _mm256_add_ps(nx, magic);
        let nb = _mm256_castps_si256(n); n = _mm256_sub_ps(n, magic);
        let mut f = _mm256_sub_ps(xc, _mm256_mul_ps(n, ln2_h)); f = _mm256_sub_ps(f, _mm256_mul_ps(n, ln2_l));
        let mut p = ec6; p = _mm256_add_ps(_mm256_mul_ps(f, p), ec5); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec4);
        p = _mm256_add_ps(_mm256_mul_ps(f, p), ec3); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec2);
        p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1);
        let pow2n = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<0>(nb), b128), 23)), _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<1>(nb), b128), 23)));
        let e2y = _mm256_mul_ps(p, pow2n);
        let tanh_v = _mm256_div_ps(_mm256_sub_ps(e2y, vone), _mm256_add_ps(e2y, vone));
        _mm256_storeu_ps(ptr, _mm256_mul_ps(vhalf, _mm256_mul_ps(x, _mm256_add_ps(vone, tanh_v))));
    }
    gelu_f32_scalar(&mut buf[n8..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn gelu_f32_sse2_inplace(buf: &mut [f32]) {
    let vk = _mm_set1_ps(0.7978845608); let vc = _mm_set1_ps(0.044715);
    let vhalf = _mm_set1_ps(0.5); let vone = _mm_set1_ps(1.0);
    let vtwo = _mm_set1_ps(2.0); let vclip = _mm_set1_ps(9.0);
    let vnclip = _mm_set1_ps(-9.0); let d256 = _mm_set1_ps(1.0/256.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i); let x = _mm_loadu_ps(ptr);
        let inner = _mm_mul_ps(vk, _mm_add_ps(x, _mm_mul_ps(vc, _mm_mul_ps(_mm_mul_ps(x, x), x))));
        let ic = _mm_min_ps(_mm_max_ps(inner, vnclip), vclip);
        let mut v = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vtwo, ic), d256), vone);
        for _ in 0..8 { v = _mm_mul_ps(v, v); }
        let tanh_v = _mm_div_ps(_mm_sub_ps(v, vone), _mm_add_ps(v, vone));
        _mm_storeu_ps(ptr, _mm_mul_ps(vhalf, _mm_mul_ps(x, _mm_add_ps(vone, tanh_v))));
    }
    gelu_f32_scalar(&mut buf[n4..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_f32_neon_inplace(buf: &mut [f32]) {
    let vk = vdupq_n_f32(0.7978845608); let vc = vdupq_n_f32(0.044715);
    let vhalf = vdupq_n_f32(0.5); let vone = vdupq_n_f32(1.0);
    let vtwo = vdupq_n_f32(2.0); let vclip = vdupq_n_f32(9.0);
    let vnclip = vdupq_n_f32(-9.0); let d256 = vdupq_n_f32(1.0/256.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i); let x = vld1q_f32(ptr);
        let inner = vmulq_f32(vk, vaddq_f32(x, vmulq_f32(vc, vmulq_f32(vmulq_f32(x, x), x))));
        let ic = vminq_f32(vmaxq_f32(inner, vnclip), vclip);
        let mut v = vfmaq_f32(vone, vmulq_f32(vtwo, ic), d256);
        for _ in 0..8 { v = vmulq_f32(v, v); }
        let tanh_v = vdivq_f32(vsubq_f32(v, vone), vaddq_f32(v, vone));
        vst1q_f32(ptr, vmulq_f32(vhalf, vmulq_f32(x, vaddq_f32(vone, tanh_v))));
    }
    gelu_f32_scalar(&mut buf[n4..]);
}

pub fn relu_i8_swar(buf: &mut [i8]) {
    let mut chunks = buf.chunks_exact_mut(8);
    for chunk in chunks.by_ref() {
        let word: u64 = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const u64) };
        let is_neg = word & 0x8080808080808080;
        let mask = (is_neg >> 7).wrapping_mul(0xFF);
        unsafe { std::ptr::write_unaligned(chunk.as_mut_ptr() as *mut u64, word & !mask); }
    }
    for b in chunks.into_remainder() { if *b < 0 { *b = 0; } }
}
