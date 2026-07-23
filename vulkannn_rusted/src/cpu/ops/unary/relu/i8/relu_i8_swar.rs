//! SWAR ReLU for I8 — the GPR-only tier (targets with 64-bit registers but no
//! vector unit). Composed from verified primitives in `cpu::swar`
//! (`relu_i8_bytes` = sign-spread mask + AND, exhaustively tested). 8 lanes per
//! u64; scalar tail. Unlike the half-float relu-select (catalog B4) there is no
//! NaN divergence — i8 has no NaN.
//!
//! BENCH: not benchmarked on x86 — this tier is never selected here (SSE4.1
//! always wins); it exists for no-SIMD targets. Correctness is the exhaustive
//! `cpu::swar` test.

use crate::cpu::swar;

#[inline]
fn load8(s: &[i8]) -> u64 {
    let mut a = [0u8; 8];
    for k in 0..8 { a[k] = s[k] as u8; }
    u64::from_le_bytes(a)
}

#[inline]
fn store8(v: u64, d: &mut [i8]) {
    for (k, b) in v.to_le_bytes().iter().enumerate() { d[k] = *b as i8; }
}

pub fn relu(in_buf: &[i8], out_buf: &mut [i8]) {
    let n = in_buf.len();
    let n8 = (n / 8) * 8;
    let mut i = 0;
    while i < n8 {
        store8(swar::relu_i8_bytes(load8(&in_buf[i..i + 8])), &mut out_buf[i..i + 8]);
        i += 8;
    }
    for j in n8..n {
        out_buf[j] = in_buf[j].max(0);
    }
}

pub fn relu_inplace(buf: &mut [i8]) {
    let n = buf.len();
    let n8 = (n / 8) * 8;
    let mut i = 0;
    while i < n8 {
        let v = swar::relu_i8_bytes(load8(&buf[i..i + 8]));
        store8(v, &mut buf[i..i + 8]);
        i += 8;
    }
    for x in buf[n8..].iter_mut() {
        *x = (*x).max(0);
    }
}
