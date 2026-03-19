#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use rayon::prelude::*;

// ReLU has been moved to src/cpu/ops/relu/

pub fn gelu_f32_inplace(buf: &mut [f32]) {
    const PAR_THRESHOLD: usize = 64_000;
    if buf.len() > PAR_THRESHOLD {
        buf.par_chunks_mut(PAR_THRESHOLD).for_each(|chunk| {
            gelu_f32_inplace_serial(chunk);
        });
        return;
    }
    gelu_f32_inplace_serial(buf);
}

fn gelu_f32_inplace_serial(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { unsafe { gelu_f32_avx_inplace(buf); } return; }
        if is_x86_feature_detected!("sse2") { unsafe { gelu_f32_sse2_inplace(buf); } return; }
    }
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

// ReLU variants moved to src/cpu/ops/relu/

pub fn gelu_f16_inplace(buf: &mut [half::f16]) {
    const CHUNK_SIZE: usize = 1024; let mut tmp = [0.0f32; CHUNK_SIZE];
    for chunk in buf.chunks_mut(CHUNK_SIZE) {
        for (i, x) in chunk.iter().enumerate() { tmp[i] = x.to_f32(); }
        gelu_f32_inplace(&mut tmp[..chunk.len()]);
        for (i, x) in chunk.iter_mut().enumerate() { *x = half::f16::from_f32(tmp[i]); }
    }
}

pub fn gelu_bf16_inplace(buf: &mut [half::bf16]) {
    const CHUNK_SIZE: usize = 1024; let mut tmp = [0.0f32; CHUNK_SIZE];
    for chunk in buf.chunks_mut(CHUNK_SIZE) {
        for (i, x) in chunk.iter().enumerate() { tmp[i] = x.to_f32(); }
        gelu_f32_inplace(&mut tmp[..chunk.len()]);
        for (i, x) in chunk.iter_mut().enumerate() { *x = half::bf16::from_f32(tmp[i]); }
    }
}

pub fn gelu_i8_dispatch(buf: &mut [i8]) {
    static mut GELU_LUT: [i8; 256] = [0; 256];
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        for i in 0..256 {
            let x = (i as i32 - 128) as f32;
            let res = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
            unsafe { GELU_LUT[i] = res.round().clamp(-128.0, 127.0) as i8; }
        }
    });
    for x in buf.iter_mut() {
        *x = unsafe { GELU_LUT[(*x as i32 + 128) as usize] };
    }
}


pub fn softmax_f32_dispatch(buf: &mut [f32], is_log: bool) {
    const PAR_THRESHOLD: usize = 65536;
    if buf.len() > PAR_THRESHOLD { softmax_f32_scalar(buf, is_log); return; }
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { unsafe { softmax_f32_avx(buf, is_log); } return; }
    }
    softmax_f32_scalar(buf, is_log);
}

pub fn softmax_f32_scalar(buf: &mut [f32], is_log: bool) {
    if buf.is_empty() { return; }
    let mut max_val = buf[0]; for &x in buf.iter() { if x > max_val { max_val = x; } }
    let mut sum = 0.0; for x in buf.iter_mut() { let val = (*x - max_val).exp(); *x = val; sum += val; }
    if is_log {
        let log_sum = sum.ln(); for x in buf.iter_mut() { *x = x.ln() - log_sum; }
    } else {
        let inv_sum = 1.0 / sum; for x in buf.iter_mut() { *x *= inv_sum; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn softmax_f32_avx(buf: &mut [f32], is_log: bool) {
    if buf.is_empty() { return; }
    let mut max_v = _mm256_set1_ps(buf[0]); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) { max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(buf.as_ptr().add(i))); }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
    let mut max_val = tmp.iter().fold(buf[0], |a, &b| a.max(b));
    for &x in &buf[n8..] { max_val = max_val.max(x); }
    let max_v = _mm256_set1_ps(max_val);
    let mut sum_v = _mm256_setzero_ps();
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i); let x = _mm256_sub_ps(_mm256_loadu_ps(ptr), max_v);
        let res = exp_f32_avx_vec(x); _mm256_storeu_ps(ptr, res); sum_v = _mm256_add_ps(sum_v, res);
    }
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut sum = tmp.iter().sum::<f32>();
    for i in n8..buf.len() { let val = (buf[i] - max_val).exp(); buf[i] = val; sum += val; }
    if is_log {
        let log_sum = sum.ln(); for i in 0..buf.len() { buf[i] = buf[i].ln() - log_sum; }
    } else {
        let inv_sum = 1.0 / sum; let inv_sum_v = _mm256_set1_ps(inv_sum);
        for i in (0..n8).step_by(8) { let ptr = buf.as_mut_ptr().add(i); _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), inv_sum_v)); }
        for i in n8..buf.len() { buf[i] *= inv_sum; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn exp_f32_avx_vec(mut x: __m256) -> __m256 {
    let log2_e = _mm256_set1_ps(1.4426950408889634); let ln2_hi = _mm256_set1_ps(0.6931457519);
    let ln2_lo = _mm256_set1_ps(1.4286067653e-6); let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-88.3762626647949); let magic = _mm256_set1_ps(12582912.0);
    let ec1 = _mm256_set1_ps(1.0); let ec2 = _mm256_set1_ps(0.5);
    let ec3 = _mm256_set1_ps(0.16666667163); let ec4 = _mm256_set1_ps(0.04166648536);
    let ec5 = _mm256_set1_ps(0.00833336077); let ec6 = _mm256_set1_ps(0.00138925374);
    let b128 = _mm_set1_epi32(127);
    x = _mm256_min_ps(x, max_x); x = _mm256_max_ps(x, min_x);
    let mut n_f = _mm256_add_ps(_mm256_mul_ps(x, log2_e), magic); let nb = _mm256_castps_si256(n_f); n_f = _mm256_sub_ps(n_f, magic);
    let f = _mm256_sub_ps(_mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)), _mm256_mul_ps(n_f, ln2_lo));
    let mut p = ec6; p = _mm256_add_ps(_mm256_mul_ps(f, p), ec5); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec4);
    p = _mm256_add_ps(_mm256_mul_ps(f, p), ec3); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec2);
    p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1);
    let pow2n = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<0>(nb), b128), 23)), _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<1>(nb), b128), 23)));
    _mm256_mul_ps(p, pow2n)
}

pub fn softmax_f16_dispatch(buf: &mut [half::f16], is_log: bool) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            unsafe { softmax_f16_f16c(buf, is_log); return; }
        }
    }
    let mut tmp: Vec<f32> = buf.iter().map(|x| x.to_f32()).collect();
    softmax_f32_scalar(&mut tmp, is_log);
    for (d, s) in buf.iter_mut().zip(tmp.iter()) { *d = half::f16::from_f32(*s); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn softmax_f16_f16c(buf: &mut [half::f16], is_log: bool) {
    if buf.is_empty() { return; }
    let mut max_v = _mm256_cvtph_ps(_mm_set1_epi16(buf[0].to_bits() as i16));
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        max_v = _mm256_max_ps(max_v, _mm256_cvtph_ps(h_vec));
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
    let mut max_val = tmp.iter().fold(buf[0].to_f32(), |a, &b| a.max(b));
    for &x in &buf[n8..] { max_val = max_val.max(x.to_f32()); }
    let max_v = _mm256_set1_ps(max_val);
    let mut sum_v = _mm256_setzero_ps();
    let mut tmp_f32 = vec![0.0f32; buf.len()];
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        let x = _mm256_sub_ps(_mm256_cvtph_ps(h_vec), max_v);
        let res = exp_f32_avx_vec(x);
        _mm256_storeu_ps(tmp_f32.as_mut_ptr().add(i), res);
        sum_v = _mm256_add_ps(sum_v, res);
    }
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut sum = tmp.iter().sum::<f32>();
    for i in n8..buf.len() { let val = (buf[i].to_f32() - max_val).exp(); tmp_f32[i] = val; sum += val; }
    if is_log {
        let log_sum = sum.ln();
        for (i, x) in tmp_f32.iter().enumerate() { buf[i] = half::f16::from_f32(x.ln() - log_sum); }
    } else {
        let inv_sum = 1.0 / sum; let inv_sum_v = _mm256_set1_ps(inv_sum);
        for i in (0..n8).step_by(8) {
            let f_vec = _mm256_loadu_ps(tmp_f32.as_ptr().add(i));
            let res_f = _mm256_mul_ps(f_vec, inv_sum_v);
            let res_h = _mm256_cvtps_ph(res_f, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(buf.as_mut_ptr().add(i) as *mut __m128i, res_h);
        }
        for i in n8..buf.len() { buf[i] = half::f16::from_f32(tmp_f32[i] * inv_sum); }
    }
}
pub fn softmax_i8_dispatch(buf: &mut [i8], is_log: bool) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") { unsafe { softmax_i8_avx2(buf, is_log); return; } }
        if is_x86_feature_detected!("avx") { unsafe { softmax_i8_avx(buf, is_log); return; } }
    }
    softmax_i8_scalar(buf, is_log);
}

pub fn softmax_i8_scalar(buf: &mut [i8], is_log: bool) {
    if buf.is_empty() { return; }
    let mut max_val = buf[0]; for &x in buf.iter() { if x > max_val { max_val = x; } }
    let mut tmp = vec![0.0f32; buf.len()];
    let mut sum = 0.0;
    for i in 0..buf.len() {
        let val = ((buf[i] as f32) - (max_val as f32)).exp();
        tmp[i] = val; sum += val;
    }
    if is_log {
        let ls = sum.ln();
        for i in 0..buf.len() { buf[i] = (tmp[i].ln() - ls) as i8; }
    } else {
        let inv_s = 1.0 / sum;
        for i in 0..buf.len() { buf[i] = (tmp[i] * inv_s) as i8; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_i8_avx2(buf: &mut [i8], is_log: bool) {
    if buf.is_empty() { return; }
    let mut max_v = _mm256_set1_epi8(buf[0]);
    let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) { max_v = _mm256_max_epi8(max_v, _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i)); }
    let mut tmp_i8 = [0i8; 32]; _mm256_storeu_si256(tmp_i8.as_mut_ptr() as *mut __m256i, max_v);
    let mut max_val = tmp_i8.iter().fold(buf[0], |a, &b| a.max(b));
    for &x in &buf[n32..] { max_val = max_val.max(x); }
    
    let max_f_v = _mm256_set1_ps(max_val as f32);
    let mut sum_v = _mm256_setzero_ps();
    let mut tmp_f32 = vec![0.0f32; buf.len()];
    
    for i in (0..n32).step_by(8) {
        let i8_data = _mm_loadl_epi64(buf.as_ptr().add(i) as *const __m128i);
        let f_v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8_data));
        let res = exp_f32_avx_vec(_mm256_sub_ps(f_v, max_f_v));
        _mm256_storeu_ps(tmp_f32.as_mut_ptr().add(i), res);
        sum_v = _mm256_add_ps(sum_v, res);
    }
    
    let mut tmp_sum = [0.0f32; 8]; _mm256_storeu_ps(tmp_sum.as_mut_ptr(), sum_v);
    let mut sum = tmp_sum.iter().sum::<f32>();
    for i in n32..buf.len() { let v = ((buf[i] as f32) - (max_val as f32)).exp(); tmp_f32[i] = v; sum += v; }
    
    if is_log {
        let ls = sum.ln();
        for i in 0..buf.len() { buf[i] = (tmp_f32[i].ln() - ls) as i8; }
    } else {
        let inv_s = 1.0 / sum; let inv_v = _mm256_set1_ps(inv_s);
        for i in (0..n32).step_by(8) {
            let f_v = _mm256_loadu_ps(tmp_f32.as_ptr().add(i));
            let res = _mm256_mul_ps(f_v, inv_v);
            // We need to pack F32 back to I8. This is slow but "dedicated".
            let mut fr = [0.0f32; 8]; _mm256_storeu_ps(fr.as_mut_ptr(), res);
            for j in 0..8 { buf[i+j] = fr[j] as i8; }
        }
        for i in n32..buf.len() { buf[i] = (tmp_f32[i] * inv_s) as i8; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn softmax_i8_avx(buf: &mut [i8], is_log: bool) {
    if buf.is_empty() { return; }
    let mut max_v = _mm_set1_epi8(buf[0]);
    let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) { max_v = _mm_max_epi8(max_v, _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i)); }
    let mut tmp_i8 = [0i8; 16]; _mm_storeu_si128(tmp_i8.as_mut_ptr() as *mut __m128i, max_v);
    let mut max_val = tmp_i8.iter().fold(buf[0], |a, &b| a.max(b));
    for &x in &buf[n16..] { max_val = max_val.max(x); }
    
    let max_f_v = _mm_set1_ps(max_val as f32);
    let mut sum_v = _mm_setzero_ps();
    let mut tmp_f32 = vec![0.0f32; buf.len()];
    
    for i in (0..n16).step_by(4) {
        let i8_data = _mm_cvtsi32_si128(*(buf.as_ptr().add(i) as *const i32));
        let f_v = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(i8_data));
        let res = exp_f32_sse_vec(_mm_sub_ps(f_v, max_f_v));
        _mm_storeu_ps(tmp_f32.as_mut_ptr().add(i), res);
        sum_v = _mm_add_ps(sum_v, res);
    }
    
    let mut tmp_sum = [0.0f32; 4]; _mm_storeu_ps(tmp_sum.as_mut_ptr(), sum_v);
    let mut sum = tmp_sum.iter().sum::<f32>();
    for i in n16..buf.len() { let v = ((buf[i] as f32) - (max_val as f32)).exp(); tmp_f32[i] = v; sum += v; }
    
    if is_log {
        let ls = sum.ln();
        for i in 0..buf.len() { buf[i] = (tmp_f32[i].ln() - ls) as i8; }
    } else {
        let inv_s = 1.0 / sum;
        for i in 0..buf.len() { buf[i] = (tmp_f32[i] * inv_s) as i8; }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn exp_f32_sse_vec(x: __m128) -> __m128 {
    let log2e = _mm_set1_ps(1.4426950408889634); let ln2h = _mm_set1_ps(0.6931457519);
    let ln2l = _mm_set1_ps(1.4286067653e-6); let magic = _mm_set1_ps(12582912.0);
    let ec1 = _mm_set1_ps(1.0); let ec2 = _mm_set1_ps(0.5);
    let ec3 = _mm_set1_ps(0.16666667163); let ec4 = _mm_set1_ps(0.04166648536);
    let ec5 = _mm_set1_ps(0.00833336077); let ec6 = _mm_set1_ps(0.00138925374);
    let b128 = _mm_set1_epi32(127);
    let mut n_f = _mm_add_ps(_mm_mul_ps(x, log2e), magic); let nb = n_f; n_f = _mm_sub_ps(n_f, magic);
    let f = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(n_f, ln2h)), _mm_mul_ps(n_f, ln2l));
    let mut p = ec6; p = _mm_add_ps(_mm_mul_ps(f, p), ec5); p = _mm_add_ps(_mm_mul_ps(f, p), ec4);
    p = _mm_add_ps(_mm_mul_ps(f, p), ec3); p = _mm_add_ps(_mm_mul_ps(f, p), ec2);
    p = _mm_add_ps(_mm_mul_ps(f, p), ec1); p = _mm_add_ps(_mm_mul_ps(f, p), ec1);
    let pow2n = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm_castps_si128(nb), b128), 23));
    _mm_mul_ps(p, pow2n)
}
