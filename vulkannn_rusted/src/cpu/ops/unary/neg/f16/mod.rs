//! F16 negation — Tier II. Legacy is scalar-only (round-trip through f32). A
//! sign-bit XOR would be faster but changes NaN payloads, so we keep the legacy
//! round-trip to preserve bit-exact parity (Rule 1).

#[inline(always)]
pub fn neg(in_buf: &[half::f16], out_buf: &mut [half::f16]) {
    for i in 0..in_buf.len() {
        out_buf[i] = half::f16::from_f32(-in_buf[i].to_f32());
    }
}

#[inline(always)]
pub fn neg_inplace(buf: &mut [half::f16]) {
    for x in buf.iter_mut() {
        *x = half::f16::from_f32(-x.to_f32());
    }
}
