/// Elementwise negation for BF16 tensors.
pub fn neg_bf16(in_buf: &[half::bf16], out_buf: &mut [half::bf16]) {
    let raw_in: &[u16] = unsafe { std::slice::from_raw_parts(in_buf.as_ptr() as *const u16, in_buf.len()) };
    let raw_out: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr() as *mut u16, out_buf.len()) };
    for (i, &x) in raw_in.iter().enumerate() { raw_out[i] = x ^ 0x8000; }
}

/// In-place negation for BF16 tensors.
pub fn neg_bf16_inplace(buf: &mut [half::bf16]) {
    let raw: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u16, buf.len()) };
    for x in raw.iter_mut() { *x ^= 0x8000; }
}
