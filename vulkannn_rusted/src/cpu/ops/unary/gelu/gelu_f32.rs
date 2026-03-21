#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elementwise GELU activation for F32 tensors with multi-architecture support.
pub fn gelu_f32(buf: &mut [f32]) {
    const PAR_THRESHOLD: usize = 64_000;
    if buf.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        buf.par_chunks_mut(PAR_THRESHOLD).for_each(|chunk| {
            gelu_f32_serial(chunk);
        });
        return;
    }
    gelu_f32_serial(buf);
}

fn gelu_f32_serial(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { 
            return unsafe { gelu_f32_avx_inplace(buf) }; 
        }
        if is_x86_feature_detected!("sse2") { 
            return unsafe { gelu_f32_sse2_inplace(buf) }; 
        }
    }

    #[cfg(target_arch = "aarch64")] {
        return gelu_f32_neon(buf);
    }

    gelu_f32_scalar(buf);
}

pub fn gelu_f32_scalar_single(v: f32) -> f32 {
    const K: f32 = 0.7978845608; 
    const C: f32 = 0.044715;
    let inner = K * (v + C * v * v * v); 
    let y = inner.clamp(-9.0, 9.0);
    let e2y = (2.0 * y).exp(); 
    let tanh_v = (e2y - 1.0) / (e2y + 1.0); 
    0.5 * v * (1.0 + tanh_v)
}

fn gelu_f32_scalar(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = gelu_f32_scalar_single(*x);
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
fn gelu_f32_neon(buf: &mut [f32]) {
    use std::arch::aarch64::*;
    // For Neon, we'll use a slightly simplified version or reuse the scalar path for now
    // as transcendental ops on Neon (exp, tanh) need a poly approx.
    gelu_f32_scalar(buf);
}
