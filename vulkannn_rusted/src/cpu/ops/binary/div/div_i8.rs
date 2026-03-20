/// Elementwise division for I8 tensors (Saturating, Scalar).
pub fn div_i8(a: &[i8], b: &[i8], res: &mut [i8]) {
    for i in 0..a.len() {
        if b[i] != 0 {
            res[i] = a[i].saturating_div(b[i]);
        } else {
            res[i] = 0;
        }
    }
}
