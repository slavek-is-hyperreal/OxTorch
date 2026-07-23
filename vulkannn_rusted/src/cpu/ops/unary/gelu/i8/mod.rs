//! I8 gelu — 256-entry LUT, transcribed VERBATIM from cpu_old (K=0.79788456,
//! clamp(-128,127)). Rule 1.
use std::sync::OnceLock;
fn lut() -> &'static [i8; 256] {
    static LUT: OnceLock<[i8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0i8; 256];
        for i in 0..256 {
            let x = (i as i32 - 128) as f32;
            let res = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
            t[i] = res.clamp(-128.0, 127.0) as i8;
        }
        t
    })
}
pub fn gelu_i8(buf: &mut [i8]) {
    let t = lut();
    for x in buf.iter_mut() { *x = t[(*x as i32 + 128) as usize]; }
}
