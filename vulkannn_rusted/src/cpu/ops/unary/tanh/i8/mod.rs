//! I8 tanh — 256-entry LUT, transcribed VERBATIM from
//! cpu_old/ops/unary/tanh/mod.rs. `x=(i-128)/16`, `tanh(x)*127`. Rule 1.

use std::sync::OnceLock;

fn lut() -> &'static [i8; 256] {
    static LUT: OnceLock<[i8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0i8; 256];
        for i in 0..256 {
            let x = (i as i32 - 128) as f32 / 16.0;
            t[i] = (x.tanh() * 127.0).round() as i8;
        }
        t
    })
}

pub fn tanh_i8(buf: &mut [i8]) {
    let t = lut();
    for x in buf.iter_mut() {
        *x = t[(*x as i32 + 128) as usize];
    }
}
