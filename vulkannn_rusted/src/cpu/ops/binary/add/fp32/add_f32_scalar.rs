//! Baseline Scalar Implementation for FP32 Addition.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.

#[inline(always)]
pub fn add(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, res.len());

    for i in 0..n {
        res[i] = a[i] + b[i];
    }
}
