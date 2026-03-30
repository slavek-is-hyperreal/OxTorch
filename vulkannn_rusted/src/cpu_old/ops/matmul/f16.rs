use matrixmultiply;
use rayon::prelude::*;

const TILE_SIZE: usize = 256;

pub fn matmul_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    let m_blocks = (m + TILE_SIZE - 1) / TILE_SIZE;
    let n_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    // Safety: each mb works on unique row range, so threads never overlap on C.
    // Get pointer OUTSIDE the parallel loop to avoid capturing the mutable reference itself.
    let c_ptr_val = c.as_mut_ptr() as usize;

    (0..m_blocks).into_par_iter().for_each(|mb| {
        let i = mb * TILE_SIZE;
        let i_end = (i + TILE_SIZE).min(m);
        let cur_m = i_end - i;

        for nb in 0..n_blocks {
            let j = nb * TILE_SIZE;
            let j_end = (j + TILE_SIZE).min(n);
            let cur_n = j_end - j;

            let mut c_tile = crate::tensor::pool::TensorPool::get_buffer::<f32>(cur_m * cur_n);
            for v in c_tile.iter_mut() { *v = 0.0; }

            for kk in (0..k).step_by(TILE_SIZE) {
                let kk_end = (kk + TILE_SIZE).min(k);
                let cur_k = kk_end - kk;

                let mut a_tile = crate::tensor::pool::TensorPool::get_buffer::<f32>(cur_m * cur_k);
                let mut b_tile = crate::tensor::pool::TensorPool::get_buffer::<f32>(cur_k * cur_n);

                // Convert A panel
                for row in 0..cur_m {
                    let a_row_src = &a[(i + row) * k + kk .. (i + row) * k + kk_end];
                    let a_row_dst = &mut a_tile[row * cur_k .. (row + 1) * cur_k];
                    for idx in 0..cur_k { a_row_dst[idx] = a_row_src[idx].to_f32(); }
                }

                // Convert B panel
                for row in 0..cur_k {
                    let b_row_src = &b[(kk + row) * n + j .. (kk + row) * n + j_end];
                    let b_row_dst = &mut b_tile[row * cur_n .. (row + 1) * cur_n];
                    for idx in 0..cur_n { b_row_dst[idx] = b_row_src[idx].to_f32(); }
                }

                unsafe {
                    matrixmultiply::sgemm(
                        cur_m, cur_k, cur_n,
                        1.0,
                        a_tile.as_ptr(), cur_k as isize, 1,
                        b_tile.as_ptr(), cur_n as isize, 1,
                        1.0,
                        c_tile.as_mut_ptr(), cur_n as isize, 1,
                    );
                }
            }

            // Write back C tile
            for row in 0..cur_m {
                unsafe {
                    let c_ptr = c_ptr_val as *mut half::f16;
                    let dst_ptr = c_ptr.add((i + row) * n + j);
                    let c_row_src = &c_tile[row * cur_n .. (row + 1) * cur_n];
                    for idx in 0..cur_n { 
                        *dst_ptr.add(idx) = half::f16::from_f32(c_row_src[idx]); 
                    }
                }
            }
        }
    });
}

pub fn linear_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    // b is (n, k), weight matrix for linear
    let m_blocks = (m + TILE_SIZE - 1) / TILE_SIZE;
    let n_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    // Safety: each mb works on unique row range
    let c_ptr_val = c.as_mut_ptr() as usize;

    (0..m_blocks).into_par_iter().for_each(|mb| {
        let i = mb * TILE_SIZE;
        let i_end = (i + TILE_SIZE).min(m);
        let cur_m = i_end - i;

        for nb in 0..n_blocks {
            let j = nb * TILE_SIZE;
            let j_end = (j + TILE_SIZE).min(n);
            let cur_n = j_end - j;

            let mut c_tile = crate::tensor::pool::TensorPool::get_buffer::<f32>(cur_m * cur_n);
            for v in c_tile.iter_mut() { *v = 0.0; }

            for kk in (0..k).step_by(TILE_SIZE) {
                let kk_end = (kk + TILE_SIZE).min(k);
                let cur_k = kk_end - kk;

                let mut a_tile = crate::tensor::pool::TensorPool::get_buffer::<f32>(cur_m * cur_k);
                let mut b_tile = crate::tensor::pool::TensorPool::get_buffer::<f32>(cur_n * cur_k); // B block is (cur_n, cur_k)

                // Convert A panel
                for row in 0..cur_m {
                    let a_row_src = &a[(i + row) * k + kk .. (i + row) * k + kk_end];
                    let a_row_dst = &mut a_tile[row * cur_k .. (row + 1) * cur_k];
                    for idx in 0..cur_k { a_row_dst[idx] = a_row_src[idx].to_f32(); }
                }

                // Convert B panel (transposed in weight storage)
                for row in 0..cur_n {
                    let b_row_src = &b[(j + row) * k + kk .. (j + row) * k + kk_end];
                    let b_row_dst = &mut b_tile[row * cur_k .. (row + 1) * cur_k];
                    for idx in 0..cur_k { b_row_dst[idx] = b_row_src[idx].to_f32(); }
                }

                unsafe {
                    matrixmultiply::sgemm(
                        cur_m, cur_k, cur_n,
                        1.0,
                        a_tile.as_ptr(), cur_k as isize, 1,
                        b_tile.as_ptr(), 1, cur_k as isize, // rs=1, cs=k for B
                        1.0,
                        c_tile.as_mut_ptr(), cur_n as isize, 1,
                    );
                }
            }

            // Write back C tile
            for row in 0..cur_m {
                unsafe {
                    let c_ptr = c_ptr_val as *mut half::f16;
                    let dst_ptr = c_ptr.add((i + row) * n + j);
                    let c_row_src = &c_tile[row * cur_n .. (row + 1) * cur_n];
                    for idx in 0..cur_n { 
                        *dst_ptr.add(idx) = half::f16::from_f32(c_row_src[idx]); 
                    }
                }
            }
        }
    });
}
