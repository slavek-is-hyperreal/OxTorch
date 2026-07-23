//! I8 sigmoid — 256-entry LUT, transcribed VERBATIM from
//! cpu_old/ops/unary/sigmoid/mod.rs. `127/(1+exp(-x/16))`, rounded. Rule 1.

use std::sync::OnceLock;

fn lut() -> &'static [i8; 256] {
    static LUT: OnceLock<[i8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0i8; 256];
        for i in 0..256 {
            let x = (i as i32 - 128) as f32;
            let res = 127.0 / (1.0 + (-x / 16.0).exp());
            t[i] = res.round() as i8;
        }
        t
    })
}

pub fn sigmoid_i8(buf: &mut [i8]) {
    let t = lut();
    for x in buf.iter_mut() {
        *x = t[(*x as i32 + 128) as usize];
    }
}
