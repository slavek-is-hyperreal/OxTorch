#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn rms_norm_f32(x: &[f32], w: &[f32], out: &mut [f32], n: usize, d: usize, eps: f32) {
    if n > 1 {
        use rayon::prelude::*;
        x.par_chunks(d).zip(out.par_chunks_mut(d)).for_each(|(x_row, out_row)| {
            rms_norm_f32_row(x_row, w, out_row, d, eps);
        });
    } else {
        rms_norm_f32_row(x, w, out, d, eps);
    }
}

fn rms_norm_f32_row(x: &[f32], w: &[f32], out: &mut [f32], d: usize, eps: f32) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") {
            return unsafe { rms_norm_f32_row_avx(x, w, out, d, eps) };
        }
    }
    rms_norm_f32_row_scalar(x, w, out, d, eps);
}

fn rms_norm_f32_row_scalar(x: &[f32], w: &[f32], out: &mut [f32], _d: usize, eps: f32) {
    let mut sq_sum = 0.0;
    for &val in x { sq_sum += val * val; }
    let rms = sq_sum / (x.len() as f32);
    let inv_rms = 1.0 / (rms + eps).sqrt();
    
    let has_w = !w.is_empty();
    for i in 0..x.len() {
        let weight = if has_w { w[i] } else { 1.0 };
        out[i] = x[i] * inv_rms * weight;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rms_norm_f32_row_avx(x: &[f32], w: &[f32], out: &mut [f32], d: usize, eps: f32) {
    let mut sq256 = _mm256_setzero_ps();
    let d8 = (d / 8) * 8;
    for i in (0..d8).step_by(8) {
        let val = _mm256_loadu_ps(x.as_ptr().add(i));
        sq256 = _mm256_add_ps(sq256, _mm256_mul_ps(val, val));
    }
    let mut sum_arr = [0.0; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sq256);
    let mut sq_sum = sum_arr.iter().sum::<f32>();
    for i in d8..d { sq_sum += x[i] * x[i]; }
    
    let mean_sq = sq_sum / (d as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let inv_rms256 = _mm256_set1_ps(inv_rms);
    
    let has_w = !w.is_empty();
    
    for i in (0..d8).step_by(8) {
        let val = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), inv_rms256);
        let weight = if has_w { _mm256_loadu_ps(w.as_ptr().add(i)) } else { _mm256_set1_ps(1.0) };
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(val, weight));
    }
    for i in d8..d {
        let weight = if has_w { w[i] } else { 1.0 };
        out[i] = x[i] * inv_rms * weight;
    }
}
