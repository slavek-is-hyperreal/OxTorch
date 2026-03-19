pub fn relu_f32(src: &[f32], dst: &mut [f32]) {
    for (o, &i) in dst.iter_mut().zip(src.iter()) {
        *o = if i > 0.0 { i } else { 0.0 };
    }
}

pub fn relu_f32_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        if *x < 0.0 { *x = 0.0; }
    }
}
