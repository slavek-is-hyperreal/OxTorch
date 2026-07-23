//! FP32 max — Tier II serial reducer. NaN ignored (legacy; docs/known_divergences §5).
pub mod max_f32_scalar;
#[cfg(target_arch = "x86_64")] pub mod max_f32_sse2;
#[cfg(target_arch = "x86_64")] pub mod max_f32_avx1;
#[cfg(target_arch = "x86_64")] pub mod max_f32_avx2;
#[cfg(target_arch = "x86_64")] pub mod max_f32_avx512;
#[cfg(target_arch = "aarch64")] pub mod max_f32_neon;
use crate::cpu::dispatch::Arch;
pub fn max(buf: &[f32], initial: f32) -> f32 {
    match crate::cpu::dispatch::active_arch() {
        #[cfg(target_arch = "x86_64")] Arch::Avx512 => unsafe { max_f32_avx512::max(buf, initial) },
        #[cfg(target_arch = "x86_64")] Arch::Avx2 => unsafe { max_f32_avx2::max(buf, initial) },
        #[cfg(target_arch = "x86_64")] Arch::Avx1 => unsafe { max_f32_avx1::max(buf, initial) },
        #[cfg(target_arch = "x86_64")] Arch::Sse2 => unsafe { max_f32_sse2::max(buf, initial) },
        #[cfg(target_arch = "aarch64")] Arch::Neon => unsafe { max_f32_neon::max(buf, initial) },
        _ => max_f32_scalar::max(buf, initial),
    }
}
#[cfg(test)]
mod t {
    use super::*;
    fn oracle(b: &[f32], i: f32) -> f32 { b.iter().fold(i, |a,&x| a.max(x)) }
    fn data(n: usize, s: u32) -> Vec<f32> { let mut st=s|1; (0..n).map(|_|{st^=st<<13;st^=st>>17;st^=st<<5;((st>>8) as f32/(1u32<<24) as f32-0.5)*100.0}).collect() }
    fn chk(f: unsafe fn(&[f32],f32)->f32) { for &n in &[0usize,1,7,8,9,17,1000,100000] { let v=data(n,n as u32+3); assert_eq!(unsafe{f(&v,f32::NEG_INFINITY)}, oracle(&v,f32::NEG_INFINITY), "n={n}"); } }
    #[test] fn scalar() { chk(|b,i| max_f32_scalar::max(b,i)); }
    #[cfg(target_arch="x86_64")] #[test] fn sse2() { if is_x86_feature_detected!("sse2") { chk(|b,i| unsafe{max_f32_sse2::max(b,i)}); } }
    #[cfg(target_arch="x86_64")] #[test] fn avx1() { if is_x86_feature_detected!("avx") { chk(|b,i| unsafe{max_f32_avx1::max(b,i)}); } }
    #[test] fn nan_ignored_like_legacy() {
        // max ignores NaN (legacy); diverges from torch which propagates.
        assert_eq!(max_f32_scalar::max(&[1.0, f32::NAN, 3.0], f32::NEG_INFINITY), 3.0);
    }
}
