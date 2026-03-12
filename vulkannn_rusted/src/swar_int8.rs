//! swar_int8.rs — GPR-based Int8 SWAR (SIMD Within A Register)
//!
//! Provides 8-way parallel arithmetic for Int8 tensors using 64-bit GPRs.
//! Targeted at legacy CPUs (Atom, Celeron, 486/Pentium style) that lack 
//! modern SIMD (SSE/AVX/NEON) but have 64-bit registers.

// use rayon::prelude::*;

/// 8-way parallel addition of two Int8 arrays using 64-bit registers.
/// Uses bit-masking to prevent carry leakage between bytes.
pub fn swar_add_int8(a: &[i8], b: &[i8], out: &mut [i8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    let n = a.len();
    let n8 = (n / 8) * 8;

    // Use GPR-based parallel add
    // Mask: 0x7F repeated 8 times = 0x7F7F7F7F7F7F7F7F
    let mask_low7 = 0x7F7F7F7F7F7F7F7F_u64;
    let mask_msb  = 0x8080808080808080_u64;

    let a_ptr = a.as_ptr() as *const u64;
    let b_ptr = b.as_ptr() as *const u64;
    let out_ptr = out.as_mut_ptr() as *mut u64;

    for i in 0..(n8/8) {
        unsafe {
            let x = *a_ptr.add(i);
            let y = *b_ptr.add(i);
            
            // SWAR Add: (x & mask) + (y & mask) handles the low 7 bits of 8 bytes.
            // The XOR with MSBs handles the final bit without leaking carry.
            let res = ((x & mask_low7) + (y & mask_low7)) ^ ((x ^ y) & mask_msb);
            *out_ptr.add(i) = res;
        }
    }

    // Scalar tail
    for i in n8..n {
        out[i] = a[i].wrapping_add(b[i]);
    }
}

/// 8-way parallel multiplication of Int8 array by a scalar.
/// Note: True SWAR multiplication is complex; this implementation uses
/// shifting for small scalars or optimized GPR unpacking.
pub fn swar_scale_int8(a: &[i8], scalar: i8, out: &mut [i8]) {
    // For now, simpler GPR unpacking for multiplication
    // (Actual SWAR mult usually needs 2 muls and masks to avoid overlap)
    let n = a.len();
    let n8 = (n / 8) * 8;
    
    let a_ptr = a.as_ptr() as *const u64;
    let out_ptr = out.as_mut_ptr() as *mut u64;
    
    let s64 = scalar as i64 as u64;

    for i in 0..(n8/8) {
        unsafe {
            let x = *a_ptr.add(i);
            // Process 4 bytes at a time to allow room for multiplication overflow
            // Low 4
            let mut res_lo = 0u64;
            for j in 0..4 {
                let byte = (x >> (j * 8)) & 0xFF;
                let mul = (byte as i8 as i64 * s64 as i8 as i64) as u8 as u64;
                res_lo |= mul << (j * 8);
            }
            // High 4
            let mut res_hi = 0u64;
            for j in 4..8 {
                let byte = (x >> (j * 8)) & 0xFF;
                let mul = (byte as i8 as i64 * s64 as i8 as i64) as u8 as u64;
                res_hi |= mul << (j * 8);
            }
            *out_ptr.add(i) = res_lo | res_hi;
        }
    }

    for i in n8..n {
        out[i] = a[i].wrapping_mul(scalar);
    }
}
