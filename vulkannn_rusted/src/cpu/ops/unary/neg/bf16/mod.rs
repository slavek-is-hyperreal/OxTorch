//! BF16 negation — Tier II. Legacy is scalar-only (round-trip through f32).

#[inline(always)]
pub fn neg(in_buf: &[half::bf16], out_buf: &mut [half::bf16]) {
    for i in 0..in_buf.len() {
        out_buf[i] = half::bf16::from_f32(-in_buf[i].to_f32());
    }
}

#[inline(always)]
pub fn neg_inplace(buf: &mut [half::bf16]) {
    for x in buf.iter_mut() {
        *x = half::bf16::from_f32(-x.to_f32());
    }
}
