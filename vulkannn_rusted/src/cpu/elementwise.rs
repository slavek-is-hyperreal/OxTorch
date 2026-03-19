#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use rayon::prelude::*;

pub fn elementwise_add_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    binop_f32(a, b, res, |va, vb| unsafe { _mm256_add_ps(va, vb) }, |x, y| x + y);
}
pub fn elementwise_sub_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    binop_f32(a, b, res, |va, vb| unsafe { _mm256_sub_ps(va, vb) }, |x, y| x - y);
}
pub fn elementwise_mul_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    binop_f32(a, b, res, |va, vb| unsafe { _mm256_mul_ps(va, vb) }, |x, y| x * y);
}
pub fn elementwise_div_f32(a: &[f32], b: &[f32], res: &mut [f32]) {
    binop_f32(a, b, res, |va, vb| unsafe { _mm256_div_ps(va, vb) }, |x, y| x / y);
}

fn binop_f32<F, S>(a: &[f32], b: &[f32], res: &mut [f32], op_simd: F, op_scalar: S) 
where F: Fn(__m256, __m256) -> __m256 + Sync + Send, S: Fn(f32, f32) -> f32 + Sync + Send 
{
    const PAR_THRESHOLD: usize = 4_000_000;
    if a.len() > PAR_THRESHOLD {
        let (a_chunks, b_chunks, res_chunks) = (a.chunks(PAR_THRESHOLD), b.chunks(PAR_THRESHOLD), res.chunks_mut(PAR_THRESHOLD));
        a_chunks.zip(b_chunks).zip(res_chunks).par_bridge().for_each(|((ac, bc), rc)| {
            binop_f32_serial(ac, bc, rc, &op_simd, &op_scalar);
        });
    } else {
        binop_f32_serial(a, b, res, &op_simd, &op_scalar);
    }
}

fn binop_f32_serial<F, S>(a: &[f32], b: &[f32], res: &mut [f32], op_simd: &F, op_scalar: &S)
where F: Fn(__m256, __m256) -> __m256, S: Fn(f32, f32) -> f32
{
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") {
            let n8 = (a.len() / 8) * 8;
            for i in (0..n8).step_by(8) {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    _mm256_storeu_ps(res.as_mut_ptr().add(i), op_simd(va, vb));
                }
            }
            for i in n8..a.len() { res[i] = op_scalar(a[i], b[i]); }
            return;
        }
    }
    for i in 0..a.len() { res[i] = op_scalar(a[i], b[i]); }
}

pub fn elementwise_op_f16(a: &[half::f16], b: &[half::f16], res: &mut [half::f16], op: &str) {
    match op {
        "add" => binop_f16(a, b, res, |va, vb| unsafe { _mm256_add_ps(va, vb) }, |x, y| x + y),
        "sub" => binop_f16(a, b, res, |va, vb| unsafe { _mm256_sub_ps(va, vb) }, |x, y| x - y),
        "mul" => binop_f16(a, b, res, |va, vb| unsafe { _mm256_mul_ps(va, vb) }, |x, y| x * y),
        "div" => binop_f16(a, b, res, |va, vb| unsafe { _mm256_div_ps(va, vb) }, |x, y| x / y),
        _ => {},
    }
}

fn binop_f16<F, S>(a: &[half::f16], b: &[half::f16], res: &mut [half::f16], op_simd: F, op_scalar: S)
where F: Fn(__m256, __m256) -> __m256 + Sync + Send, S: Fn(f32, f32) -> f32 + Sync + Send
{
    const PAR_THRESHOLD: usize = 256_000;
    if a.len() > PAR_THRESHOLD {
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            binop_f16_serial(ac, bc, rc, &op_simd, &op_scalar);
        });
    } else {
        binop_f16_serial(a, b, res, &op_simd, &op_scalar);
    }
}

fn binop_f16_serial<F, S>(a: &[half::f16], b: &[half::f16], res: &mut [half::f16], op_simd: &F, op_scalar: &S)
where F: Fn(__m256, __m256) -> __m256, S: Fn(f32, f32) -> f32
{
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            let n8 = (a.len() / 8) * 8;
            for i in (0..n8).step_by(8) {
                unsafe {
                    let va = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
                    let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
                    let vr = op_simd(va, vb);
                    _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT));
                }
            }
            for i in n8..a.len() { res[i] = half::f16::from_f32(op_scalar(a[i].to_f32(), b[i].to_f32())); }
            return;
        }
    }
    for i in 0..a.len() { res[i] = half::f16::from_f32(op_scalar(a[i].to_f32(), b[i].to_f32())); }
}

pub fn elementwise_op_bf16(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16], op: &str) {
    match op {
        "add" => binop_bf16(a, b, res, |va, vb| unsafe { _mm256_add_ps(va, vb) }, |x, y| x + y),
        "sub" => binop_bf16(a, b, res, |va, vb| unsafe { _mm256_sub_ps(va, vb) }, |x, y| x - y),
        "mul" => binop_bf16(a, b, res, |va, vb| unsafe { _mm256_mul_ps(va, vb) }, |x, y| x * y),
        "div" => binop_bf16(a, b, res, |va, vb| unsafe { _mm256_div_ps(va, vb) }, |x, y| x / y),
        _ => {},
    }
}

fn binop_bf16<F, S>(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16], op_simd: F, op_scalar: S)
where F: Fn(__m256, __m256) -> __m256 + Sync + Send, S: Fn(f32, f32) -> f32 + Sync + Send
{
    const PAR_THRESHOLD: usize = 256_000;
    if a.len() > PAR_THRESHOLD {
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            binop_bf16_serial(ac, bc, rc, &op_simd, &op_scalar);
        });
    } else {
        binop_bf16_serial(a, b, res, &op_simd, &op_scalar);
    }
}

fn binop_bf16_serial<F, S>(a: &[half::bf16], b: &[half::bf16], res: &mut [half::bf16], op_simd: &F, op_scalar: &S)
where F: Fn(__m256, __m256) -> __m256, S: Fn(f32, f32) -> f32
{
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") {
            let n8 = (a.len() / 8) * 8;
            let zero = unsafe { _mm_setzero_si128() };
            for i in (0..n8).step_by(8) {
                unsafe {
                    let ba = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                    let bb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                    let va = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_unpacklo_epi16(zero, ba)), _mm_unpackhi_epi16(zero, ba)));
                    let vb = _mm256_castsi256_ps(_mm256_insertf128_si256::<1>(_mm256_castsi128_si256(_mm_unpacklo_epi16(zero, bb)), _mm_unpackhi_epi16(zero, bb)));
                    let vr = op_simd(va, vb);
                    let res_si = _mm256_castps_si256(vr);
                    let h_lo = _mm_srli_epi32(_mm256_castsi256_si128(res_si), 16);
                    let h_hi = _mm_srli_epi32(_mm256_extractf128_si256(res_si, 1), 16);
                    _mm_storeu_si128(res.as_mut_ptr().add(i) as *mut __m128i, _mm_packus_epi32(h_lo, h_hi));
                }
            }
            for i in n8..a.len() { res[i] = half::bf16::from_f32(op_scalar(a[i].to_f32(), b[i].to_f32())); }
            return;
        }
    }
    for i in 0..a.len() { res[i] = half::bf16::from_f32(op_scalar(a[i].to_f32(), b[i].to_f32())); }
}
pub fn elementwise_op_i8(a: &[i8], b: &[i8], res: &mut [i8], op: &str) {
    match op {
        "add" => binop_i8(a, b, res, |va, vb| unsafe { _mm256_adds_epi8(va, vb) }, |x, y| x.saturating_add(y)),
        "sub" => binop_i8(a, b, res, |va, vb| unsafe { _mm256_subs_epi8(va, vb) }, |x, y| x.saturating_sub(y)),
        "mul" => binop_i8_mul(a, b, res),
        _ => {},
    }
}

fn binop_i8<F, S>(a: &[i8], b: &[i8], res: &mut [i8], op_simd: F, op_scalar: S)
where F: Fn(__m256i, __m256i) -> __m256i + Sync + Send, S: Fn(i8, i8) -> i8 + Sync + Send
{
    const PAR_THRESHOLD: usize = 4_000_000;
    if a.len() > PAR_THRESHOLD {
        a.chunks(PAR_THRESHOLD).zip(b.chunks(PAR_THRESHOLD)).zip(res.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|((ac, bc), rc)| {
            binop_i8_serial(ac, bc, rc, &op_simd, &op_scalar);
        });
    } else {
        binop_i8_serial(a, b, res, &op_simd, &op_scalar);
    }
}

fn binop_i8_serial<F, S>(a: &[i8], b: &[i8], res: &mut [i8], op_simd: &F, op_scalar: &S)
where F: Fn(__m256i, __m256i) -> __m256i, S: Fn(i8, i8) -> i8
{
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            let n32 = (a.len() / 32) * 32;
            for i in (0..n32).step_by(32) {
                unsafe {
                    let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                    let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                    _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, op_simd(va, vb));
                }
            }
            for i in n32..a.len() { res[i] = op_scalar(a[i], b[i]); }
            return;
        }
    }
    for i in 0..a.len() { res[i] = op_scalar(a[i], b[i]); }
}

fn binop_i8_mul(a: &[i8], b: &[i8], res: &mut [i8]) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            let n32 = (a.len() / 32) * 32;
            for i in (0..n32).step_by(32) {
                unsafe {
                    let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                    let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                    // Unpack to i16 for multiplication
                    let lo_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
                    let hi_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
                    let lo_b = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
                    let hi_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
                    
                    let prod_lo = _mm256_mullo_epi16(lo_a, lo_b);
                    let prod_hi = _mm256_mullo_epi16(hi_a, hi_b);
                    
                    // Pack back with saturation to i8
                    let res_vec = _mm256_packs_epi16(prod_lo, prod_hi);
                    // Fix permute: packs_epi16 produces (lo[0..7], hi[0..7], lo[8..15], hi[8..15])
                    let res_ordered = _mm256_permute4x64_epi64(res_vec, 0xD8); 
                    _mm256_storeu_si256(res.as_mut_ptr().add(i) as *mut __m256i, res_ordered);
                }
            }
            for i in n32..a.len() { res[i] = a[i].saturating_mul(b[i]); }
            return;
        }
    }
    for i in 0..a.len() { res[i] = a[i].saturating_mul(b[i]); }
}
