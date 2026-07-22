//! Tensor ⊕ scalar broadcast (`scalar_op_*`). MOVE-not-rewrite from
//! cpu_old/ops/binary/scalar.rs: the op is selected by a `&str` ("add"/"sub"/
//! "mul"/"div"), which does not fit the per-arch file matrix, so it is
//! transcribed as one module (like matmul/bitnet). SIMD kept inline exactly as
//! legacy. Only change: the f32 rayon threshold now reads cpu::thresholds
//! (ScalarOpF32, default 1_000_000 = legacy value) instead of a hardcoded const.
//!
//! Rayon lives in this {op}-level module only (Tier III). Numeric quirk (Rule 6):
//! div-by-zero returns 0.0/0 exactly as legacy.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::thresholds::{self, Threshold};

/// Scalar operations for F32 tensors.
pub fn scalar_op_f32(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    let t = thresholds::get(Threshold::ScalarOpF32);
    if in_buf.len() > t {
        use rayon::prelude::*;
        in_buf
            .chunks(t)
            .zip(out_buf.chunks_mut(t))
            .par_bridge()
            .for_each(|(ic, oc)| scalar_op_f32_serial(ic, scalar, oc, op));
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
        }
        "sub" => {
            for i in (0..n8).step_by(8) {
                let va = _mm256_loadu_ps(in_buf.as_ptr().add(i));
                _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_sub_ps(va, vs));
            }
        }
        "mul" => {
            for i in (0..n8).step_by(8) {
                let va = _mm256_loadu_ps(in_buf.as_ptr().add(i));
                _mm256_storeu_ps(out_buf.as_mut_ptr().add(i), _mm256_mul_ps(va, vs));
            }
        }
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
        }
        _ => {}
    }
    scalar_op_f32_scalar(&in_buf[n8..], scalar, &mut out_buf[n8..], op);
}

fn scalar_op_f32_scalar(in_buf: &[f32], scalar: f32, out_buf: &mut [f32], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x + scalar; },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x - scalar; },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x * scalar; },
        "div" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = if scalar != 0.0 { x / scalar } else { 0.0 }; },
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
        "div" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = if scalar != 0.0 { half::f16::from_f32(x.to_f32() / scalar) } else { half::f16::ZERO }; },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

/// Scalar operations for BF16 tensors (scalar path — legacy has no BF16 SIMD).
pub fn scalar_op_bf16(in_buf: &[half::bf16], scalar: f32, out_buf: &mut [half::bf16], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() + scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() - scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = half::bf16::from_f32(x.to_f32() * scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = if scalar != 0.0 { half::bf16::from_f32(x.to_f32() / scalar) } else { half::bf16::ZERO }; },
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
        }
        "sub" => {
            for i in (0..n16).step_by(16) {
                let va = _mm_loadu_si128(in_buf.as_ptr().add(i) as *const __m128i);
                _mm_storeu_si128(out_buf.as_mut_ptr().add(i) as *mut __m128i, _mm_subs_epi8(va, vs));
            }
        }
        _ => return scalar_op_i8_scalar(in_buf, scalar, out_buf, op),
    }
    scalar_op_i8_scalar(&in_buf[n16..], scalar, &mut out_buf[n16..], op);
}

fn scalar_op_i8_scalar(in_buf: &[i8], scalar: i8, out_buf: &mut [i8], op: &str) {
    match op {
        "add" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_add(scalar); },
        "sub" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_sub(scalar); },
        "mul" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = x.saturating_mul(scalar); },
        "div" => for (i, &x) in in_buf.iter().enumerate() { out_buf[i] = if scalar != 0 { x / scalar } else { 0 }; },
        _ => out_buf.copy_from_slice(in_buf),
    }
}

// Move-not-rewrite: bit-exact vs legacy scalar.rs for every op × dtype.
#[cfg(test)]
mod parity {
    fn cmp_f32(op: &str, scalar: f32) {
        for &n in &[1usize, 7, 8, 9, 17, 1023, 65_536] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 8.0).collect();
            let (mut g, mut w) = (vec![0f32; n], vec![0f32; n]);
            super::scalar_op_f32(&a, scalar, &mut g, op);
            crate::cpu_old::ops::binary::scalar::scalar_op_f32(&a, scalar, &mut w, op);
            assert_eq!(g, w, "scalar_op_f32 {op} s={scalar} n={n}");
        }
    }
    fn cmp_i8(op: &str, scalar: i8) {
        for &n in &[1usize, 15, 16, 17, 33, 1023] {
            let a: Vec<i8> = (0..n).map(|i| (i as i32 - 64) as i8).collect();
            let (mut g, mut w) = (vec![0i8; n], vec![0i8; n]);
            super::scalar_op_i8(&a, scalar, &mut g, op);
            crate::cpu_old::ops::binary::scalar::scalar_op_i8(&a, scalar, &mut w, op);
            assert_eq!(g, w, "scalar_op_i8 {op} s={scalar} n={n}");
        }
    }
    #[test] fn scalar_f32_vs_legacy() { for op in ["add","sub","mul","div"] { cmp_f32(op, 3.5); cmp_f32(op, 0.0); } }
    #[test] fn scalar_i8_vs_legacy() { for op in ["add","sub","mul","div"] { cmp_i8(op, 5); cmp_i8(op, 0); } }
}
