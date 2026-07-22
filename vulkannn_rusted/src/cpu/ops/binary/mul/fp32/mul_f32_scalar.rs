//! Baseline Scalar Implementation for FP32 Multiply.
//! Part of the OxTorch Scientific-Grade Specialization Matrix.
//! Semantic reference — transcribed from cpu_old/ops/binary/mul/mul_f32.rs.

#[inline(always)]
pub fn mul(a: &[f32], b: &[f32], res: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, res.len());
    for i in 0..n {
        res[i] = a[i] * b[i];
    }
}
