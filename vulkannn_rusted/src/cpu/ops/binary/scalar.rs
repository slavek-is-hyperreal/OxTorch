#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar operations for F32 tensors.
pub fn scalar_op_f32(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    const PAR_THRESHOLD: usize = 1_000_000;
    if in_buf.len() > PAR_THRESHOLD {
        use rayon::prelude::*;
        in_buf.chunks(PAR_THRESHOLD).zip(out_buf.chunks_mut(PAR_THRESHOLD)).par_bridge().for_each(|(ic, oc)| {
            scalar_op_f32_serial(ic, scalar, oc, op);
        });
    } else {
        scalar_op_f32_serial(in_buf, scalar, out_buf, op);
    }
}

fn scalar_op_f32_serial(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { scalar_op_f32_avx2(in_buf, scalar, out_buf, op) };
        }
    }
    scalar_op_f32_scalar(in_buf, scalar, out_buf, op);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scalar_op_f32_avx2(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    let n8 = (in_buf.len() / 8) * 8;
    let vs = _mm256_set1_ps(scalar);
    
    match op {
        "add" => {
            for i in (0..n8).step_by(8) {
                let va = _mm256_loadu_ps(in_buf.as_ptr().add(i));
                _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_add_ps(va, vs));
            }
        },
        "sub" => {
            for i in (0..n8).step_by(8) {
                let va = _mm256_loadu_ps(in_buf.as_ptr().add(i));
                _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_sub_ps(va, vs));
            }
        },
        "mul" => {
            for i in (0..n8).step_by(8) {
                let va = _mm256_loadu_ps(in_buf.as_ptr().add(i));
                _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_mul_ps(va, vs));
            }
        },
        "div" => {
            if scalar != 0.0 {
                for i in (0..n8).step_by(8) {
                    let va = _mm256_loadu_ps(in_buf.as_ptr().add(i));
                    _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_div_ps(va, vs));
                }
            } else {
                for i in (0..n8).step_by(8) {
                    _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_setzero_ps());
                }
            }
        },
        _ => {}
    }
    scalar_op_f32_scalar(&in_buf[n8..], scalar, &mut out_buf[n8..], op);
}

fn scalar_op_f32_scalar(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x + scalar; },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x - scalar; },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x * scalar; },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0.0 { out_buf[i] = x / scalar; } else { out_buf[i] = 0.0; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for F16 tensors.
pub fn scalar_op_f16(in_buf: &[half::f16], scalar: f32, out_buf: &mut [half::f16], op: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { scalar_op_f16_f16c(in_buf, scalar, out_buf, op) };
        }
    }
    scalar_op_f16_scalar(in_buf, scalar, out_buf, op);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn scalar_op_f16_f16c(in_buf: &[half::f16], scalar: f32, out_buf: &mut [half::f16], op: &str) {
    let n8 = (in_buf.len() / 8) * 8;
    let vs = _mm256_set1_ps(scalar);
    for i in (0..n8).step_by(8) {
        let va = _mm256_cvtph_ps(_mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i));
        let vr = match op {
            "add" => _mm256_add_ps(va, vs),
            "sub" => _mm256_sub_ps(va, vs),
            "mul" => _mm256_mul_ps(va, vs),
            "div" => if scalar != 0.0 { _mm256_div_ps(va, vs) } else { _mm256_setzero_ps() },
            _ => va,
        };
        _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT));
    }
    scalar_op_f16_scalar(&in_buf[n8..], scalar, &mut out_buf[n8..], op);
}

fn scalar_op_f16_scalar(in_buf: &[half::f16], scalar: f32, out_buf: &mut [half::f16], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32() + scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32() - scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::f16::from_f32(x.to_f32() * scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0.0 { out_buf[i] = half::f16::from_f32(x.to_f32() / scalar); } else { out_buf[i] = half::f16::ZERO; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for BF16 tensors.
pub fn scalar_op_bf16(in_buf: &[half::bf16], scalar: f32, out_buf: &mut [half::bf16], op: &str) {
    // For BF16 we don't have good SIMD conversion in AVX2 (only in AVX512), keep scalar for now or use manual bit manipulations.
    // Simplifying: use scalar.
    scalar_op_bf16_scalar(in_buf, scalar, out_buf, op);
}

fn scalar_op_bf16_scalar(in_buf: &[half::bf16], scalar: f32, out_buf: &mut [half::bf16], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() + scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() - scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() * scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0.0 { out_buf[i] = half::bf16::from_f32(x.to_f32() / scalar); } else { out_buf[i] = half::bf16::ZERO; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for I8 tensors.
pub fn scalar_op_i8(in_buf: &[i8], scalar: i8, out_buf: &mut [i8], op: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { scalar_op_i8_sse2(in_buf, scalar, out_buf, op) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    scalar_op_i8_scalar(in_buf, scalar, out_buf, op);
}

#[cfg(target_arch = "x86_64")]
unsafe fn scalar_op_i8_sse2(in_buf: &[i8], scalar: i8, out_buf: &mut [i8], op: &str) {
    let n16 = (in_buf.len() / 16) * 16;
    let vs = _mm_set1_epi8(scalar);
    match op {
        "add" => {
            for i in (0..n16).step_by(16) {
                let va = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
                _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, _mm_adds_epi8(va, vs));
            }
        },
        "sub" => {
            for i in (0..n16).step_by(16) {
                let va = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
                _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, _mm_subs_epi8(va, vs));
            }
        },
        // Mul/Div for i8 are less common for saturating, keep as scalar for now.
        _ => {
             return scalar_op_i8_scalar(in_buf, scalar, out_buf, op);
        }
    }
    scalar_op_i8_scalar(&in_buf[n16..], scalar, &mut out_buf[n16..], op);
}

fn scalar_op_i8_scalar(in_buf: &[i8], scalar: i8, out_buf: &mut [i8], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_add(scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_sub(scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_mul(scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { if scalar != 0 { out_buf[i] = x / scalar; } else { out_buf[i] = 0; } },
        _ => out_buf.copy_from_slice(in_buf),
    }
}
