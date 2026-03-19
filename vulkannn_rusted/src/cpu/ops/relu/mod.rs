mod generic;
mod avx;
mod f16;
mod bf16;
mod i8;

pub use generic::{relu_f32 as generic_f32, relu_f32_inplace as generic_f32_inplace};
#[cfg(target_arch = "x86_64")]
pub use avx::{relu_f32 as avx_f32, relu_f32_inplace as avx_f32_inplace};

pub use f16::{relu_f16, relu_f16_inplace};
pub use bf16::{relu_bf16, relu_bf16_inplace};
pub use i8::{relu_i8_inplace, relu_i8_swar};

use rayon::prelude::*;

const PAR_THRESHOLD: usize = 128_000;

pub fn relu_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    if src.len() > PAR_THRESHOLD {
        dst.par_chunks_mut(PAR_THRESHOLD)
           .zip(src.par_chunks(PAR_THRESHOLD))
           .for_each(|(d_chunk, s_chunk)| {
               dispatch_relu_f32(s_chunk, d_chunk);
           });
        return;
    }
    dispatch_relu_f32(src, dst);
}

pub fn relu_f32_inplace(buf: &mut [f32]) {
    if buf.len() > PAR_THRESHOLD {
        buf.par_chunks_mut(PAR_THRESHOLD).for_each(|chunk| {
            dispatch_relu_f32_inplace(chunk);
        });
        return;
    }
    dispatch_relu_f32_inplace(buf);
}

#[inline(always)]
fn dispatch_relu_f32(src: &[f32], dst: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    unsafe { return avx::relu_f32(src, dst); }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
    generic::relu_f32(src, dst);
}

#[inline(always)]
fn dispatch_relu_f32_inplace(buf: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    unsafe { return avx::relu_f32_inplace(buf); }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
    generic::relu_f32_inplace(buf);
}
