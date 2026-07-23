//! RMS_NORM — Tier III (rayon over rows). Per-row RMS normalize (f32 accum, std
//! sqrt). f16/bf16 convert-through-f32.
pub mod fp32;
use rayon::prelude::*;
#[inline]
fn row_f32(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    { if is_x86_feature_detected!("avx") { return unsafe { fp32::rms_avx1::row(x, w, out, eps) }; } }
    fp32::rms_scalar::row(x, w, out, eps);
}
pub fn rms_norm_f32(x: &[f32], w: &[f32], out: &mut [f32], n: usize, d: usize, eps: f32) {
    if n > 1 { x.par_chunks(d).zip(out.par_chunks_mut(d)).for_each(|(xr, or)| row_f32(xr, w, or, eps)); }
    else { row_f32(x, w, out, eps); }
}
macro_rules! rms_half {
    ($half:ty, $fn:ident) => {
        pub fn $fn(x: &[$half], w: &[$half], out: &mut [$half], n: usize, d: usize, eps: f32) {
            let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
            let do_row = |xr: &[$half], or: &mut [$half]| {
                let xf: Vec<f32> = xr.iter().map(|v| v.to_f32()).collect();
                let mut of = vec![0f32; xr.len()];
                row_f32(&xf, &wf, &mut of, eps);
                for (d2, &s) in or.iter_mut().zip(of.iter()) { *d2 = <$half>::from_f32(s); }
            };
            if n > 1 { x.par_chunks(d).zip(out.par_chunks_mut(d)).for_each(|(xr, or)| do_row(xr, or)); }
            else { do_row(x, out); }
        }
    };
}
rms_half!(half::f16, rms_norm_f16);
rms_half!(half::bf16, rms_norm_bf16);
