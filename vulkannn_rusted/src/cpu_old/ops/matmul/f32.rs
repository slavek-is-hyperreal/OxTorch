use matrixmultiply;

pub fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.0,
            c.as_mut_ptr(), n as isize, 1
        );
    }
}

pub fn linear_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), 1, k as isize,
            0.0,
            c.as_mut_ptr(), n as isize, 1
        );
    }
}
