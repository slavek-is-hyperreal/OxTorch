#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn layer_norm_f32(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], n: usize, d: usize, eps: f32) {
    if n > 1 {
        use rayon::prelude::*;
        x.par_chunks(d).zip(out.par_chunks_mut(d)).for_each(|(x_row, out_row)| {
            layer_norm_f32_row(x_row, w, b, out_row, d, eps);
        });
    } else {
        layer_norm_f32_row(x, w, b, out, d, eps);
    }
}

fn layer_norm_f32_row(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], d: usize, eps: f32) {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx") {
            return unsafe { layer_norm_f32_row_avx(x, w, b, out, d, eps) };
        }
    }
    layer_norm_f32_row_scalar(x, w, b, out, d, eps);
}

fn layer_norm_f32_row_scalar(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], _d: usize, eps: f32) {
    let mut sum = 0.0;
    for &val in x { sum += val; }
    let mean = sum / (x.len() as f32);
    
    let mut var_sum = 0.0;
    for &val in x { 
        let diff = val - mean;
        var_sum += diff * diff; 
    }
    let var = var_sum / (x.len() as f32);
    let inv_std = 1.0 / (var + eps).sqrt();
    
    let has_w = !w.is_empty();
    let has_b = !b.is_empty();

    for i in 0..x.len() {
        let weight = if has_w { w[i] } else { 1.0 };
        let bias = if has_b { b[i] } else { 0.0 };
        out[i] = (x[i] - mean) * inv_std * weight + bias;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn layer_norm_f32_row_avx(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], d: usize, eps: f32) {
    let mut sum256 = _mm256_setzero_ps();
    let d8 = (d / 8) * 8;
    for i in (0..d8).step_by(8) {
        sum256 = _mm256_add_ps(sum256, _mm256_loadu_ps(x.as_ptr().add(i)));
    }
    let mut sum_arr = [0.0; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum256);
    let mut sum = sum_arr.iter().sum::<f32>();
    for i in d8..d { sum += x[i]; }
    let mean = sum / (d as f32);
    
    let mean256 = _mm256_set1_ps(mean);
    let mut var256 = _mm256_setzero_ps();
    for i in (0..d8).step_by(8) {
        let val = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean256);
        var256 = _mm256_add_ps(var256, _mm256_mul_ps(val, val));
    }
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), var256);
    let mut var = sum_arr.iter().sum::<f32>();
    for i in d8..d { 
        let val = x[i] - mean;
        var += val * val;
    }
    var /= d as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let inv_std256 = _mm256_set1_ps(inv_std);
    
    let has_w = !w.is_empty();
    let has_b = !b.is_empty();
    
    for i in (0..d8).step_by(8) {
        let val = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean256), inv_std256);
        let weight = if has_w { _mm256_loadu_ps(w.as_ptr().add(i)) } else { _mm256_set1_ps(1.0) };
        let bias = if has_b { _mm256_loadu_ps(b.as_ptr().add(i)) } else { _mm256_setzero_ps() };
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(_mm256_mul_ps(val, weight), bias));
    }
    for i in d8..d {
        let weight = if has_w { w[i] } else { 1.0 };
        let bias = if has_b { b[i] } else { 0.0 };
        out[i] = (x[i] - mean) * inv_std * weight + bias;
    }
}
