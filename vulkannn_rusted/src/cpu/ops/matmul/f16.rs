use half;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
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

    c.fill(half::f16::ZERO);

    c.par_chunks_mut(block_m * n).enumerate().for_each(|(mi_idx, c_m_tile)| {
        let mi = mi_idx * block_m;
        let m_this_tile = (m - mi).min(block_m);
        
        for ki in (0..k).step_by(block_k) {
            let k_end = (ki + block_k).min(k);
            for ni in (0..n).step_by(block_n) {
                let n_end = (ni + block_n).min(n);

                for i_rel in (0..m_this_tile).step_by(6) {
                    let i = mi + i_rel;
                    let i_rows = (m_this_tile - i_rel).min(6);
                    for j in (ni..n_end).step_by(8) {
                        if n_end - j < 8 {
                             // Tail
                             for ii in 0..i_rows {
                                 for kk in ki..k_end {
                                     let aval = a[(i + ii) * k + kk].to_f32();
                                     for jj in j..n_end {
                                         let mut res = c_m_tile[(i_rel+ii)*n + jj].to_f32();
                                         res += aval * b[kk*n + jj].to_f32();
                                         c_m_tile[(i_rel+ii)*n + jj] = half::f16::from_f32(res);
                                     }
                                 }
                             }
                             continue;
                        }

                        let mut accs = [_mm256_setzero_ps(); 6];
                        for ii in 0..i_rows {
                            let curr_c = &c_m_tile[(i_rel+ii)*n + j .. (i_rel+ii)*n + j + 8];
                            accs[ii] = _mm256_cvtph_ps(_mm_loadu_si128(curr_c.as_ptr() as *const __m128i));
                        }

                        for kk in ki..k_end {
                            let b_vec = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(kk * n + j) as *const __m128i));
                            for ii in 0..i_rows {
                                let a_vec = _mm256_set1_ps(a[(i + ii) * k + kk].to_f32());
                                accs[ii] = _mm256_add_ps(accs[ii], _mm256_mul_ps(a_vec, b_vec));
                            }
                        }

                        for ii in 0..i_rows {
                            let c_bits = _mm256_cvtps_ph(accs[ii], _MM_FROUND_TO_NEAREST_INT);
                            _mm_storeu_si128(c_m_tile.as_mut_ptr().add((i_rel+ii)*n + j) as *mut __m128i, c_bits);
                        }
                    }
                }
            }
        }
    });
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma", target_feature = "f16c"))]
unsafe fn matmul_f16_avx2_fma_f16c(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    // Exact same optimization as before, using fmadd
    let block_m = 64;
    c.fill(half::f16::ZERO);

    c.par_chunks_mut(block_m * n).enumerate().for_each(|(mi_idx, c_m_tile)| {
        let mi = mi_idx * block_m;
        let m_this_tile = (m - mi).min(block_m);
        
        for ki in (0..k).step_by(128) {
            let k_end = (ki + 128).min(k);
            for ni in (0..n).step_by(64) {
                let n_end = (ni + 64).min(n);

                for i_rel in (0..m_this_tile).step_by(4) {
                    let i = mi + i_rel;
                    let i_rows = (m_this_tile - i_rel).min(4);
                    for j in (ni..n_end).step_by(16) {
                        if n_end - j < 16 {
                             for ii in 0..i_rows {
                                 for kk in ki..k_end {
                                     let aval = a[(i + ii) * k + kk].to_f32();
                                     for jj in j..n_end {
                                         let mut res = c_m_tile[(i_rel+ii)*n + jj].to_f32();
                                         res += aval * b[kk*n + jj].to_f32();
                                         c_m_tile[(i_rel+ii)*n + jj] = half::f16::from_f32(res);
                                     }
                                 }
                             }
                             continue;
                        }

                        let mut accs = [[_mm256_setzero_ps(); 2]; 4];
                        for ii in 0..i_rows {
                            for jj_off in 0..2 {
                                let curr_c = &c_m_tile[(i_rel+ii)*n + j + jj_off*8 .. (i_rel+ii)*n + j + jj_off*8 + 8];
                                accs[ii][jj_off] = _mm256_cvtph_ps(_mm_loadu_si128(curr_c.as_ptr() as *const __m128i));
                            }
                        }

                        for kk in ki..k_end {
                            let b_vec0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(kk * n + j) as *const __m128i));
                            let b_vec1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(kk * n + j + 8) as *const __m128i));
                            for ii in 0..i_rows {
                                let a_vec = _mm256_set1_ps(a[(i + ii) * k + kk].to_f32());
                                accs[ii][0] = _mm256_fmadd_ps(a_vec, b_vec0, accs[ii][0]);
                                accs[ii][1] = _mm256_fmadd_ps(a_vec, b_vec1, accs[ii][1]);
                            }
                        }

                        for ii in 0..i_rows {
                            for jj_off in 0..2 {
                                let res_bits = _mm256_cvtps_ph(accs[ii][jj_off], _MM_FROUND_TO_NEAREST_INT);
                                _mm_storeu_si128(c_m_tile.as_mut_ptr().add((i_rel+ii)*n + j + jj_off*8) as *mut __m128i, res_bits);
                            }
                        }
                    }
                }
            }
        }
    });
}

pub fn matmul_f16_scalar(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    c.fill(half::f16::ZERO);
    for i in 0..m {
        for kk in 0..k {
            let aval = a[i * k + kk].to_f32();
            for j in 0..n {
                let mut res = c[i * n + j].to_f32();
                res += aval * b[kk * n + j].to_f32();
                c[i * n + j] = half::f16::from_f32(res);
            }
        }
    }
}

pub fn linear_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "f16c"))]
    {
        unsafe { linear_f16_avx1_f16c(m, k, n, a, b, c); return; }
    }
    
    c.fill(half::f16::ZERO);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk].to_f32() * b[j * k + kk].to_f32();
            }
            c[i * n + j] = half::f16::from_f32(sum);
        }
    }
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
