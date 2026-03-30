pub fn sub_f32_scalar(a: &[f32], b: &[f32], res: &mut [f32]) {
    for i in 0..a.len() {
        res[i] = a[i] - b[i];
    }
}
