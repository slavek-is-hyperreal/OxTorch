use matrixmultiply;

pub fn matmul_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    let mut c_f32 = vec![0.0f32; m * n];

    for i in 0..a.len() { a_f32[i] = a[i].to_f32(); }
    for i in 0..b.len() { b_f32[i] = b[i].to_f32(); }

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a_f32.as_ptr(), k as isize, 1,
            b_f32.as_ptr(), n as isize, 1,
            0.0,
            c_f32.as_mut_ptr(), n as isize, 1,
        );
    }

    for i in 0..c.len() { c[i] = half::f16::from_f32(c_f32[i]); }
}

pub fn linear_f16(m: usize, k: usize, n: usize, a: &[half::f16], b: &[half::f16], c: &mut [half::f16]) {
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; n * k];
    let mut c_f32 = vec![0.0f32; m * n];

    for i in 0..a.len() { a_f32[i] = a[i].to_f32(); }
    for i in 0..b.len() { b_f32[i] = b[i].to_f32(); }

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a_f32.as_ptr(), k as isize, 1,
            b_f32.as_ptr(), 1, k as isize,
            0.0,
            c_f32.as_mut_ptr(), n as isize, 1,
        );
    }

    for i in 0..c.len() { c[i] = half::f16::from_f32(c_f32[i]); }
}
