/// Softmax reduction for I8 tensors (pseudo-softmax).
pub fn softmax_i8(buf: &mut [i8], _is_log: bool) {
    // Softmax on I8 is rare; providing a simple saturating version
    if buf.is_empty() { return; }
    let max_val = buf.iter().fold(i8::MIN, |a, &b| a.max(b));
    for x in buf.iter_mut() {
        if *x == max_val { *x = 127; } else { *x = -128; } // Hardmax for i8
    }
}
