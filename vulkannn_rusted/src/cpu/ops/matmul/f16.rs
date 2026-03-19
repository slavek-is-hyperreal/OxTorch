use half;
#[allow(unused_imports)]
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

pub fn matmul_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
    {
        #[cfg(target_feature = "fma")]
        unsafe { matmul_f16_avx2_fma_f16c(m, k, n, a, b, c); return; }
        
        unsafe { matmul_f16_avx1_f16c(m, k, n, a, b, c); return; }
    }

    // Scalar fallback
    matmul_f16_scalar(m, k, n, a, b, c);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
unsafe fn matmul_f16_avx1_f16c(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    let block_m = 64;
    let block_n = 512;
    let block_k = 128;

    c.par_chunks_mut(block_m * n).enumerate().for_each(|(mi_idx, c_m_tile)| {
        let mi = mi_idx * block_m;
        let m_this_tile = (m - mi).min(block_m);
        
        // Intermediate f32 accumulator for the whole row-block
        let mut acc_f32 = vec![0.0f32; m_this_tile * n];

        for ki in (0..k).step_by(block_k) {
            let k_end = (ki + block_k).min(k);
            
            for i_rel in (0..m_this_tile).step_by(6) {
                let i = mi + i_rel;
                let i_rows = (m_this_tile - i_rel).min(6);
                
                for j in (0..n).step_by(8) {
                    let mut accs = [_mm256_setzero_ps(); 6];
                    // Load current f32 partial sums
                    for ii in 0..i_rows {
                        if n - j >= 8 {
                            accs[ii] = _mm256_loadu_ps(acc_f32.as_ptr().add(ii_rel_to_idx(i_rel + ii, j, n)));
                        }
                    }

                    for kk in ki..k_end {
                        let b_ptr = b.as_ptr().add(kk * n + j);
                        let b_vec = if n - j >= 8 {
                            _mm256_cvtph_ps(_mm_loadu_si128(b_ptr as *const __m128i))
                        } else {
                            let mut tmp = [half::f16::ZERO; 8];
                            for jj in 0..(n-j) { tmp[jj] = *b_ptr.add(jj); }
                            _mm256_cvtph_ps(_mm_loadu_si128(tmp.as_ptr() as *const __m128i))
                        };

                        for ii in 0..i_rows {
                            let a_vec = _mm256_set1_ps(a[(i + ii) * k + kk].to_f32());
                            accs[ii] = _mm256_add_ps(accs[ii], _mm256_mul_ps(a_vec, b_vec));
                        }
                    }

                    // Store back to f32 accumulator
                    for ii in 0..i_rows {
                        if n - j >= 8 {
                            _mm256_storeu_ps(acc_f32.as_mut_ptr().add(ii_rel_to_idx(i_rel + ii, j, n)), accs[ii]);
                        } else {
                            let mut tmp = [0.0f32; 8];
                            _mm256_storeu_ps(tmp.as_mut_ptr(), accs[ii]);
                            for jj in 0..(n-j) { acc_f32[ii_rel_to_idx(i_rel + ii, j + jj, n)] = tmp[jj]; }
                        }
                    }
                }
            }
        }

        // Final convert to f16
        for i_rel in 0..m_this_tile {
            for j in (0..n).step_by(8) {
                let f_ptr = acc_f32.as_ptr().add(i_rel * n + j);
                let f_vec = if n - j >= 8 {
                    _mm256_loadu_ps(f_ptr)
                } else {
                    let mut tmp = [0.0f32; 8];
                    for jj in 0..(n-j) { tmp[jj] = *f_ptr.add(jj); }
                    _mm256_loadu_ps(tmp.as_ptr())
                };
                let h_bits = _mm256_cvtps_ph(f_vec, _MM_FROUND_TO_NEAREST_INT);
                let c_ptr = c_m_tile.as_mut_ptr().add(i_rel * n + j);
                if n - j >= 8 {
                    _mm_storeu_si128(c_ptr as *mut __m128i, h_bits);
                } else {
                    let h_tmp: [half::f16; 8] = core::mem::transmute(h_bits);
                    for jj in 0..(n-j) { *c_ptr.add(jj) = h_tmp[jj]; }
                }
            }
        }
    });
}

#[inline(always)]
fn ii_rel_to_idx(i_rel: usize, j: usize, n: usize) -> usize {
    i_rel * n + j
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma", target_feature = "f16c"))]
unsafe fn matmul_f16_avx2_fma_f16c(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    let block_m = 64;
    let block_k = 128;

    c.par_chunks_mut(block_m * n).enumerate().for_each(|(mi_idx, c_m_tile)| {
        let mi = mi_idx * block_m;
        let m_this_tile = (m - mi).min(block_m);
        
        let mut acc_f32 = vec![0.0f32; m_this_tile * n];

        for ki in (0..k).step_by(block_k) {
            let k_end = (ki + block_k).min(k);
            
            for i_rel in (0..m_this_tile).step_by(4) {
                let i = mi + i_rel;
                let i_rows = (m_this_tile - i_rel).min(4);
                
                for j in (0..n).step_by(16) {
                    let mut accs = [[_mm256_setzero_ps(); 2]; 4];
                    
                    for ii in 0..i_rows {
                        for jj_off in 0..2 {
                            let curr_j = j + jj_off * 8;
                            if curr_j < n {
                                if n - curr_j >= 8 {
                                    accs[ii][jj_off] = _mm256_loadu_ps(acc_f32.as_ptr().add(ii_rel_to_idx(i_rel + ii, curr_j, n)));
                                }
                            }
                        }
                    }

                    for kk in ki..k_end {
                        let b0_ptr = b.as_ptr().add(kk * n + j);
                        let b_vec0 = if n - j >= 8 {
                            _mm256_cvtph_ps(_mm_loadu_si128(b0_ptr as *const __m128i))
                        } else {
                            let mut tmp = [half::f16::ZERO; 8];
                            for jj in 0..(n-j) { tmp[jj] = *b0_ptr.add(jj); }
                            _mm256_cvtph_ps(_mm_loadu_si128(tmp.as_ptr() as *const __m128i))
                        };

                        let b_vec1 = if n - j >= 16 {
                            _mm256_cvtph_ps(_mm_loadu_si128(b0_ptr.add(8) as *const __m128i))
                        } else if n - j > 8 {
                            let mut tmp = [half::f16::ZERO; 8];
                            for jj in 0..(n-j-8) { tmp[jj] = *b0_ptr.add(8+jj); }
                            _mm256_cvtph_ps(_mm_loadu_si128(tmp.as_ptr() as *const __m128i))
                        } else {
                            _mm256_setzero_ps()
                        };

                        for ii in 0..i_rows {
                            let a_vec = _mm256_set1_ps(a[(i + ii) * k + kk].to_f32());
                            accs[ii][0] = _mm256_fmadd_ps(a_vec, b_vec0, accs[ii][0]);
                            accs[ii][1] = _mm256_fmadd_ps(a_vec, b_vec1, accs[ii][1]);
                        }
                    }

                    for ii in 0..i_rows {
                        for jj_off in 0..2 {
                            let curr_j = j + jj_off * 8;
                            if curr_j < n {
                                if n - curr_j >= 8 {
                                    _mm256_storeu_ps(acc_f32.as_mut_ptr().add(ii_rel_to_idx(i_rel + ii, curr_j, n)), accs[ii][jj_off]);
                                } else {
                                    let mut tmp = [0.0f32; 8];
                                    _mm256_storeu_ps(tmp.as_mut_ptr(), accs[ii][jj_off]);
                                    for jj in 0..(n - curr_j) { acc_f32[ii_rel_to_idx(i_rel + ii, curr_j + jj, n)] = tmp[jj]; }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to f16
        for i_rel in 0..m_this_tile {
            for j in (0..n).step_by(8) {
                let f_ptr = acc_f32.as_ptr().add(i_rel * n + j);
                let f_vec = if n - j >= 8 {
                    _mm256_loadu_ps(f_ptr)
                } else {
                    let mut tmp = [0.0f32; 8];
                    for jj in 0..(n-j) { tmp[jj] = *f_ptr.add(jj); }
                    _mm256_loadu_ps(tmp.as_ptr())
                };
                let h_bits = _mm256_cvtps_ph(f_vec, _MM_FROUND_TO_NEAREST_INT);
                let c_ptr = c_m_tile.as_mut_ptr().add(i_rel * n + j);
                if n - j >= 8 {
                    _mm_storeu_si128(c_ptr as *mut __m128i, h_bits);
                } else {
                    let h_tmp: [half::f16; 8] = core::mem::transmute(h_bits);
                    for jj in 0..(n-j) { *c_ptr.add(jj) = h_tmp[jj]; }
                }
            }
        }
    });
}

pub fn matmul_f16_scalar(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    c.par_chunks_mut(n).enumerate().for_each(|(i, row_c)| {
        let mut acc = vec![0.0f32; n];
        for kk in 0..k {
            let aval = a[i * k + kk].to_f32();
            if aval == 0.0 { continue; }
            for j in 0..n {
                acc[j] += aval * b[kk * n + j].to_f32();
            }
        }
        for j in 0..n {
            row_c[j] = half::f16::from_f32(acc[j]);
        }
    });
}

pub fn linear_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
    {
        unsafe { linear_f16_avx1_f16c(m, k, n, a, b, c); return; }
    }
    
    c.par_chunks_mut(n).enumerate().for_each(|(i, row_c)| {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk].to_f32() * b[j * k + kk].to_f32();
            }
            row_c[j] = half::f16::from_f32(sum);
        }
    });
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
unsafe fn linear_f16_avx1_f16c(_m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    c.fill(half::f16::ZERO);
    c.par_chunks_mut(n).enumerate().for_each(|(i, row_c)| {
        for j in 0..n {
            let mut sum_v = _mm256_setzero_ps();
            let mut kk = 0;
            while kk + 8 <= k {
                let a_bits = _mm_loadu_si128(a.as_ptr().add(i * k + kk) as *const __m128i);
                let a_v = _mm256_cvtph_ps(a_bits);
                let b_bits = _mm_loadu_si128(b.as_ptr().add(j * k + kk) as *const __m128i);
                let b_v = _mm256_cvtph_ps(b_bits);
                sum_v = _mm256_add_ps(sum_v, _mm256_mul_ps(a_v, b_v));
                kk += 8;
            }
            
            let x128 = _mm_add_ps(_mm256_extractf128_ps(sum_v, 0), _mm256_extractf128_ps(sum_v, 1));
            let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
            let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 1));
            let mut sum = _mm_cvtss_f32(x32);
            
            while kk < k {
                sum += a[i * k + kk].to_f32() * b[j * k + kk].to_f32();
                kk += 1;
            }
            row_c[j] = half::f16::from_f32(sum);
        }
    });
}
