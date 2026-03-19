#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use rayon::prelude::*;

pub fn exp_f32_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { unsafe { exp_f32_avx_inplace(buf); } return; }
        if is_x86_feature_detected!("sse2") { unsafe { exp_f32_sse2_inplace(buf); } return; }
    }
    #[cfg(target_arch = "aarch64")] { unsafe { exp_f32_neon_inplace(buf); } return; }
    buf.par_iter_mut().for_each(|x| *x = x.exp());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn exp_f32_avx_inplace(buf: &mut [f32]) {
    let log2_e = _mm256_set1_ps(1.4426950408889634); let ln2_hi = _mm256_set1_ps(0.6931457519);
    let ln2_lo = _mm256_set1_ps(1.4286067653e-6); let max_x = _mm256_set1_ps(88.3762626647949);
    let min_x = _mm256_set1_ps(-88.3762626647949); let magic = _mm256_set1_ps(12582912.0);
    let ec1 = _mm256_set1_ps(1.0); let ec2 = _mm256_set1_ps(0.5);
    let ec3 = _mm256_set1_ps(0.16666667163); let ec4 = _mm256_set1_ps(0.04166648536);
    let ec5 = _mm256_set1_ps(0.00833336077); let ec6 = _mm256_set1_ps(0.00138925374);
    let b128 = _mm_set1_epi32(127);
    let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let ptr = buf.as_mut_ptr().add(i); let mut x = _mm256_min_ps(_mm256_loadu_ps(ptr), max_x); x = _mm256_max_ps(x, min_x);
        let mut n_f = _mm256_add_ps(_mm256_mul_ps(x, log2_e), magic); let nb = _mm256_castps_si256(n_f); n_f = _mm256_sub_ps(n_f, magic);
        let f = _mm256_sub_ps(_mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)), _mm256_mul_ps(n_f, ln2_lo));
        let mut p = ec6; p = _mm256_add_ps(_mm256_mul_ps(f, p), ec5); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec4);
        p = _mm256_add_ps(_mm256_mul_ps(f, p), ec3); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec2);
        p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1); p = _mm256_add_ps(_mm256_mul_ps(f, p), ec1);
        let pow2n = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<0>(nb), b128), 23)), _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256::<1>(nb), b128), 23)));
        _mm256_storeu_ps(ptr, _mm256_mul_ps(p, pow2n));
    }
    for x in buf[n8..].iter_mut() { *x = x.exp(); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn exp_f32_sse2_inplace(buf: &mut [f32]) {
    let d256 = _mm_set1_ps(1.0/256.0); let one = _mm_set1_ps(1.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i); let mut v = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ptr), d256), one);
        for _ in 0..8 { v = _mm_mul_ps(v, v); } _mm_storeu_ps(ptr, v);
    }
    for x in buf[n4..].iter_mut() { *x = x.exp(); }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_f32_neon_inplace(buf: &mut [f32]) {
    let d256 = vdupq_n_f32(1.0/256.0); let one = vdupq_n_f32(1.0);
    let n4 = (buf.len() / 4) * 4;
    for i in (0..n4).step_by(4) {
        let ptr = buf.as_mut_ptr().add(i); let mut v = vfmaq_f32(one, vld1q_f32(ptr), d256);
        for _ in 0..8 { v = vmulq_f32(v, v); } vst1q_f32(ptr, v);
    }
    for x in buf[n4..].iter_mut() { *x = x.exp(); }
}

pub fn sum_f16_dispatch(buf: &[half::f16]) -> f32 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_f16_dispatch_serial(chunk)).sum();
    }
    sum_f16_dispatch_serial(buf)
}

fn sum_f16_dispatch_serial(buf: &[half::f16]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { sum_f16_f16c(buf) };
        }
    }
    buf.iter().map(|x| x.to_f32()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn sum_f16_f16c(buf: &[half::f16]) -> f32 {
    let mut sum_v = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        sum_v = _mm256_add_ps(sum_v, _mm256_cvtph_ps(h_vec));
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().sum::<f32>(); for &x in &buf[n8..] { s += x.to_f32(); } s
}

pub fn sum_bf16_dispatch(buf: &[half::bf16]) -> f32 {
    const PAR_LIMIT: usize = 128_000;
    if buf.len() > PAR_LIMIT {
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_bf16_dispatch_serial(chunk)).sum();
    }
    sum_bf16_dispatch_serial(buf)
}

fn sum_bf16_dispatch_serial(buf: &[half::bf16]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { sum_bf16_avx(buf) }; }
    }
    buf.iter().map(|x| x.to_f32()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_bf16_avx(buf: &[half::bf16]) -> f32 {
    let mut sum_v = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    let zero128 = _mm_setzero_si128();
    for i in (0..n8).step_by(8) {
        let b_raw = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, b_raw)),
            _mm_unpackhi_epi16(zero128, b_raw)
        ));
        sum_v = _mm256_add_ps(sum_v, f_vec);
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().sum::<f32>(); for &x in &buf[n8..] { s += x.to_f32(); } s
}

pub fn sum_f32_dispatch(buf: &[f32]) -> f32 {
    const PAR_LIMIT: usize = 256_000;
    if buf.len() > PAR_LIMIT {
        return buf.par_chunks(PAR_LIMIT).map(|chunk| sum_f32_serial(chunk)).sum();
    }
    sum_f32_serial(buf)
}

fn sum_f32_serial(buf: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        unsafe {
            if is_x86_feature_detected!("avx512f") { return sum_f32_avx512(buf); }
            if is_x86_feature_detected!("avx") { return sum_f32_avx(buf); }
        }
    }
    buf.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f32_avx512(buf: &[f32]) -> f32 {
    let mut sum_v = _mm512_setzero_ps(); let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) { sum_v = _mm512_add_ps(sum_v, _mm512_loadu_ps(buf.as_ptr().add(i))); }
    let mut tmp = [0.0f32; 16]; _mm512_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().sum::<f32>(); for &x in &buf[n16..] { s += x; } s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_f32_avx(buf: &[f32]) -> f32 {
    let mut sum_v = _mm256_setzero_ps(); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) { sum_v = _mm256_add_ps(sum_v, _mm256_loadu_ps(buf.as_ptr().add(i))); }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), sum_v);
    let mut s = tmp.iter().sum::<f32>(); for &x in &buf[n8..] { s += x; } s
}

pub fn sum_i8_dispatch(buf: &[i8]) -> i64 {
    const PAR_LIMIT: usize = 256_000;
    if buf.len() > PAR_LIMIT { return buf.par_chunks(PAR_LIMIT).map(|c| sum_i8_serial(c) as i64).sum(); }
    sum_i8_serial(buf)
}

fn sum_i8_serial(buf: &[i8]) -> i64 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx512f") { return unsafe { sum_i8_avx512(buf) }; }
        if is_x86_feature_detected!("avx2") { return unsafe { sum_i8_avx2(buf) }; }
        if is_x86_feature_detected!("avx") { return unsafe { sum_i8_avx1(buf) }; }
    }
    sum_i8_swar(buf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_i8_avx512(buf: &[i8]) -> i64 {
    let mut sum_v = _mm512_setzero_si512(); let n64 = (buf.len() / 64) * 64;
    for i in (0..n64).step_by(64) {
        let v = _mm512_loadu_si512(buf.as_ptr().add(i) as *const __m512i);
        let lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v));
        let hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(v, 1));
        sum_v = _mm512_add_epi32(sum_v, _mm512_add_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(lo)), _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(lo, 1))));
        sum_v = _mm512_add_epi32(sum_v, _mm512_add_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(hi)), _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(hi, 1))));
    }
    let s = _mm512_reduce_add_epi32(sum_v); 
    let mut s_acc = s as i64; for &b in &buf[n64..] { s_acc += b as i64; } s_acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_i8_avx2(buf: &[i8]) -> i64 {
    let mut sum_v = _mm256_setzero_si256(); let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) {
        let v = _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i);
        let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
        let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
        sum_v = _mm256_add_epi32(sum_v, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo, 1))));
        sum_v = _mm256_add_epi32(sum_v, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi, 1))));
    }
    let mut tmp = [0i32; 8]; _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum_v);
    let mut s = tmp.iter().map(|&x| x as i64).sum::<i64>(); for &b in &buf[n32..] { s += b as i64; } s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_i8_avx1(buf: &[i8]) -> i64 {
    let mut sum_v = _mm_setzero_si128(); let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) {
        let v = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        let lo = _mm_cvtepi8_epi16(v);
        let hi = _mm_cvtepi8_epi16(_mm_srli_si128(v, 8));
        sum_v = _mm_add_epi32(sum_v, _mm_add_epi32(_mm_cvtepi16_epi32(lo), _mm_cvtepi16_epi32(_mm_srli_si128(lo, 8))));
        sum_v = _mm_add_epi32(sum_v, _mm_add_epi32(_mm_cvtepi16_epi32(hi), _mm_cvtepi16_epi32(_mm_srli_si128(hi, 8))));
    }
    let mut tmp = [0i32; 4]; _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum_v);
    let mut s = tmp.iter().map(|&x| x as i64).sum::<i64>(); for &b in &buf[n16..] { s += b as i64; } s
}

pub fn sum_i8_swar(buf: &[i8]) -> i64 {
    buf.iter().map(|&x| x as i64).sum()
}

pub fn max_i8_dispatch(buf: &[i8], initial: i8) -> i8 {
    if buf.len() > 128_000 { return buf.par_chunks(128_000).map(|c| max_i8_dispatch_serial(c, initial)).reduce(|| initial, |a, b| a.max(b)); }
    max_i8_dispatch_serial(buf, initial)
}

fn max_i8_dispatch_serial(buf: &[i8], initial: i8) -> i8 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") { return unsafe { max_i8_avx2(buf, initial) }; }
        if is_x86_feature_detected!("avx") { return unsafe { max_i8_avx1(buf, initial) }; }
    }
    buf.iter().fold(initial, |a, &b| a.max(b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_i8_avx2(buf: &[i8], initial: i8) -> i8 {
    let mut max_v = _mm256_set1_epi8(initial); let n32 = (buf.len() / 32) * 32;
    for i in (0..n32).step_by(32) { max_v = _mm256_max_epi8(max_v, _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i)); }
    let mut tmp = [0i8; 32]; _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n32..] { m = m.max(x); } m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_i8_avx1(buf: &[i8], initial: i8) -> i8 {
    let mut max_v = _mm_set1_epi8(initial); let n16 = (buf.len() / 16) * 16;
    for i in (0..n16).step_by(16) { max_v = _mm_max_epi8(max_v, _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i)); }
    let mut tmp = [0i8; 16]; _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n16..] { m = m.max(x); } m
}

pub fn max_f16_dispatch(buf: &[half::f16], initial: f32) -> f32 {
    if buf.len() > 128_000 { return buf.par_chunks(128_000).map(|c| max_f16_dispatch_serial(c, initial)).reduce(|| initial, |a, b| a.max(b)); }
    max_f16_dispatch_serial(buf, initial)
}

fn max_f16_dispatch_serial(buf: &[half::f16], initial: f32) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { max_f16_f16c(buf, initial) };
        }
    }
    buf.iter().fold(initial, |a, &b| a.max(b.to_f32()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn max_f16_f16c(buf: &[half::f16], initial: f32) -> f32 {
    let mut max_v = _mm256_set1_ps(initial); let n8 = (buf.len() / 8) * 8;
    for i in (0..n8).step_by(8) {
        let h_vec = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        max_v = _mm256_max_ps(max_v, _mm256_cvtph_ps(h_vec));
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n8..] { m = m.max(x.to_f32()); } m
}

pub fn max_bf16_dispatch(buf: &[half::bf16], initial: f32) -> f32 {
    if buf.len() > 128_000 { return buf.par_chunks(128_000).map(|c| max_bf16_dispatch_serial(c, initial)).reduce(|| initial, |a, b| a.max(b)); }
    max_bf16_dispatch_serial(buf, initial)
}

fn max_bf16_dispatch_serial(buf: &[half::bf16], initial: f32) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") { return unsafe { max_bf16_avx(buf, initial) }; }
    }
    buf.iter().fold(initial, |a, &b| a.max(b.to_f32()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_bf16_avx(buf: &[half::bf16], initial: f32) -> f32 {
    let mut max_v = _mm256_set1_ps(initial); let n8 = (buf.len() / 8) * 8;
    let zero128 = _mm_setzero_si128();
    for i in (0..n8).step_by(8) {
        let b_raw = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
        let f_vec = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi16(zero128, b_raw)),
            _mm_unpackhi_epi16(zero128, b_raw)
        ));
        max_v = _mm256_max_ps(max_v, f_vec);
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), max_v);
    let mut m = tmp.iter().fold(initial, |a, &b| a.max(b));
    for &x in &buf[n8..] { m = m.max(x.to_f32()); } m
}

pub fn softmax_bf16_dispatch(buf: &mut [half::bf16], is_log: bool) {
    if buf.is_empty() { return; }
    let max_val = max_bf16_dispatch(buf, f32::NEG_INFINITY);
    let sum: f32 = buf.par_chunks(64_000).map(|chunk| {
        let mut s = 0.0f32;
        for x in chunk { s += (x.to_f32() - max_val).exp(); }
        s
    }).sum();
    
    if is_log {
        let log_sum = sum.ln();
        buf.par_chunks_mut(64_000).for_each(|chunk| {
            for x in chunk { *x = half::bf16::from_f32(x.to_f32() - max_val - log_sum); }
        });
    } else {
        let inv_sum = 1.0 / sum;
        buf.par_chunks_mut(64_000).for_each(|chunk| {
            for x in chunk { *x = half::bf16::from_f32((x.to_f32() - max_val).exp() * inv_sum); }
        });
    }
}
