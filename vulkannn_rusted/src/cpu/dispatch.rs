//! dispatch.rs — Tier II/III scaffolding for CPU kernels.
//!
//! This module owns two things and nothing else:
//!
//! 1. **`Arch` + the runtime override** (`force_arch` / `active_arch`). The
//!    override exists so that an AVX-capable box can still be made to execute the
//!    SSE2 or scalar path — needed both for honest per-variant benchmarking and
//!    for bisecting a numerical divergence between two implementations of the
//!    same kernel.
//! 2. **The `elementwise_binary!` / `elementwise_unary!` macros**, which generate
//!    *only* scaffolding: the `is_x86_feature_detected!` ladder, the rayon gate
//!    reading [`crate::cpu::thresholds`], chunking and tail handling.
//!
//! The macros deliberately do **not** synthesise SIMD loops. Every vector loop
//! lives, hand-written, in its own `[op]_[dtype]_[arch].rs` file; the macro is
//! handed the set of functions that exist for a given op and wires them up.
//!
//! # Why runtime detection and not `#[cfg(target_feature)]`
//! `src/cpu/ops/binary/add/fp32/mod.rs` is the cautionary example: it selects its
//! kernel with `#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]`. On a
//! stock `cargo build` (no `-C target-cpu=native`) *none* of those cfgs are active,
//! so a machine with AVX silently executes the scalar fallback. `sub/` gets this
//! right by testing `is_x86_feature_detected!` at runtime, and that is the pattern
//! these macros encode.

use std::sync::atomic::{AtomicU8, Ordering};

/// Instruction-set variant a kernel can be compiled/dispatched for.
///
/// `Swar` is the "SIMD within a register" fallback: portable integer-register
/// tricks used when no vector unit is available (or when it is not worth the
/// transition penalty).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Arch {
    Scalar = 0,
    Swar = 1,
    Sse2 = 2,
    Avx1 = 3,
    Avx2 = 4,
    Avx512 = 5,
    Neon = 6,
}

/// Sentinel stored in [`FORCED_ARCH`] meaning "no override, use detection".
const NO_OVERRIDE: u8 = u8::MAX;

static FORCED_ARCH: AtomicU8 = AtomicU8::new(NO_OVERRIDE);

impl Arch {
    /// Capability rank **within one target family**. Higher == wider/newer.
    ///
    /// Ranks are only comparable after [`Arch::family_matches_target`] has
    /// confirmed both values belong to this build's family; `Neon` and `Avx1`
    /// can never be compared because they never coexist.
    #[inline]
    pub const fn rank(self) -> u8 {
        match self {
            Arch::Scalar => 0,
            Arch::Swar => 1,
            Arch::Sse2 => 2,
            Arch::Avx1 => 3,
            Arch::Avx2 => 4,
            Arch::Avx512 => 5,
            // aarch64 baseline vector unit; ranks above Swar in its own family.
            Arch::Neon => 2,
        }
    }

    /// Stable lowercase name (`"avx1"`, `"neon"`, …). Used by `force_arch_by_name`.
    pub const fn name(self) -> &'static str {
        match self {
            Arch::Scalar => "scalar",
            Arch::Swar => "swar",
            Arch::Sse2 => "sse2",
            Arch::Avx1 => "avx1",
            Arch::Avx2 => "avx2",
            Arch::Avx512 => "avx512",
            Arch::Neon => "neon",
        }
    }

    /// Parse a lowercase arch name; `None` for anything unrecognised.
    pub fn from_name(s: &str) -> Option<Arch> {
        match s {
            "scalar" => Some(Arch::Scalar),
            "swar" => Some(Arch::Swar),
            "sse2" => Some(Arch::Sse2),
            "avx1" | "avx" => Some(Arch::Avx1),
            "avx2" => Some(Arch::Avx2),
            "avx512" | "avx512f" => Some(Arch::Avx512),
            "neon" => Some(Arch::Neon),
            _ => None,
        }
    }

    fn from_u8(v: u8) -> Option<Arch> {
        match v {
            0 => Some(Arch::Scalar),
            1 => Some(Arch::Swar),
            2 => Some(Arch::Sse2),
            3 => Some(Arch::Avx1),
            4 => Some(Arch::Avx2),
            5 => Some(Arch::Avx512),
            6 => Some(Arch::Neon),
            _ => None,
        }
    }

    /// Whether this variant can exist at all on the target we were compiled for.
    /// Prevents `force_arch(Some(Arch::Neon))` from selecting NEON on x86_64.
    #[inline]
    pub const fn family_matches_target(self) -> bool {
        match self {
            Arch::Scalar | Arch::Swar => true,
            Arch::Sse2 | Arch::Avx1 | Arch::Avx2 | Arch::Avx512 => cfg!(target_arch = "x86_64"),
            Arch::Neon => cfg!(target_arch = "aarch64"),
        }
    }
}

/// Pin every subsequent dispatch to `arch`, or pass `None` to restore normal
/// runtime detection.
///
/// # Safety of over-forcing
/// Forcing an arch the CPU does not actually implement is **not** honoured: the
/// request is clamped down to what [`detect_arch`] reports. Forcing *down*
/// (e.g. `Sse2` on an AVX machine) always works and is the intended use.
///
/// This is a process-global switch. Tests that use it must not run concurrently
/// with benchmarks or with each other on the same op.
pub fn force_arch(arch: Option<Arch>) {
    let v = match arch {
        Some(a) => a as u8,
        None => NO_OVERRIDE,
    };
    FORCED_ARCH.store(v, Ordering::Relaxed);
}

/// Convenience wrapper over [`force_arch`] taking a name (`"sse2"`, `"scalar"`, …).
/// Returns `false` if the name is unknown.
pub fn force_arch_by_name(name: &str) -> bool {
    match Arch::from_name(name) {
        Some(a) => {
            force_arch(Some(a));
            true
        }
        None => false,
    }
}

/// The currently requested override, if any. Not clamped — see [`active_arch`].
#[inline]
pub fn forced_arch() -> Option<Arch> {
    Arch::from_u8(FORCED_ARCH.load(Ordering::Relaxed))
}

/// The best variant this CPU actually supports, ignoring any override.
#[inline]
pub fn detect_arch() -> Arch {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 leaf kernels enable f/dq/vl (+fma); gate on all of them so we
        // never dispatch to a kernel using an instruction the CPU lacks. Any
        // AVX-512-foundation CPU (Skylake-X onward) provides the whole set.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("fma")
        {
            return Arch::Avx512;
        }
        // Every avx2 leaf kernel in this crate enables "avx2,fma"; gate the rung
        // on both so a rare AVX2-without-FMA CPU never reaches an FMA kernel.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Arch::Avx2;
        }
        if is_x86_feature_detected!("avx") {
            return Arch::Avx1;
        }
        if is_x86_feature_detected!("sse2") {
            return Arch::Sse2;
        }
        return Arch::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Arch::Neon;
        }
        return Arch::Scalar;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Arch::Scalar
    }
}

/// The variant that dispatch ladders will actually select.
///
/// Equals [`detect_arch`] unless an override is in force *and* that override is
/// both same-family and no wider than what the CPU supports.
#[inline]
pub fn active_arch() -> Arch {
    let detected = detect_arch();
    match forced_arch() {
        Some(f) if f.family_matches_target() && f.rank() <= detected.rank() => f,
        // Unsatisfiable request (wrong family / wider than the CPU): ignore it
        // rather than execute an illegal instruction.
        Some(_) => detected,
        None => detected,
    }
}

/// Run `body` with the arch override pinned to `arch`, restoring the previous
/// override afterwards. Intended for `#[cfg(test)]` and benchmark use.
pub fn with_forced_arch<R>(arch: Option<Arch>, body: impl FnOnce() -> R) -> R {
    let prev = forced_arch();
    force_arch(arch);
    let out = body();
    force_arch(prev);
    out
}

/// Generate the Tier II runtime dispatch ladder + Tier III rayon gate for an
/// **elementwise binary** kernel `f(&[T], &[T], &mut [T])`.
///
/// The macro emits exactly one public function. It does **not** emit any SIMD.
///
/// ```ignore
/// elementwise_binary! {
///     name: sub_f32,
///     elem: f32,
///     threshold: crate::cpu::thresholds::Threshold::SubF32,
///     scalar: crate::cpu::ops::binary::sub::fp32::sub_f32_scalar::sub_f32_scalar,
///     sse2:   crate::cpu::ops::binary::sub::fp32::sub_f32_sse2::sub_f32_sse2,
///     avx1:   crate::cpu::ops::binary::sub::fp32::sub_f32_avx1::sub_f32_avx1,
///     avx2:   crate::cpu::ops::binary::sub::fp32::sub_f32_avx2::sub_f32_avx2,
/// }
/// ```
///
/// Field contract:
/// * `name`, `elem`, `threshold`, `scalar` are **mandatory**, in that order.
/// * `swar`, `sse2`, `avx1`, `avx2`, `neon` are optional and must appear in that
///   order after `scalar`. Omit the ones the op does not implement; the ladder
///   falls through to the next-lower variant that *is* supplied.
/// * `scalar` and `swar` must be **safe** `fn`s. `sse2`/`avx1`/`avx2`/`neon` must
///   be `unsafe fn`s (they are invoked inside an `unsafe` block after the
///   corresponding runtime feature check has passed).
/// * Slice lengths are asserted equal up front; each kernel may assume
///   `a.len() == b.len() == res.len()`.
#[macro_export]
macro_rules! elementwise_binary {
    (
        name: $name:ident,
        elem: $elem:ty,
        threshold: $threshold:expr,
        scalar: $scalar:path,
        $( swar: $swar:path, )?
        $( sse2: $sse2:path, )?
        $( avx1: $avx1:path, )?
        $( avx2: $avx2:path, )?
        $( neon: $neon:path, )?
    ) => {
        pub fn $name(a: &[$elem], b: &[$elem], res: &mut [$elem]) {
            // ---- Tier II: runtime instruction-set ladder -------------------
            #[inline]
            fn __serial(a: &[$elem], b: &[$elem], res: &mut [$elem]) {
                #[allow(unused_variables)]
                let arch = $crate::cpu::dispatch::active_arch();

                #[cfg(target_arch = "x86_64")]
                {
                    $(
                        if arch.rank() >= $crate::cpu::dispatch::Arch::Avx2.rank() {
                            return unsafe { $avx2(a, b, res) };
                        }
                    )?
                    $(
                        if arch.rank() >= $crate::cpu::dispatch::Arch::Avx1.rank() {
                            return unsafe { $avx1(a, b, res) };
                        }
                    )?
                    $(
                        if arch.rank() >= $crate::cpu::dispatch::Arch::Sse2.rank() {
                            return unsafe { $sse2(a, b, res) };
                        }
                    )?
                }

                #[cfg(target_arch = "aarch64")]
                {
                    $(
                        if arch.rank() >= $crate::cpu::dispatch::Arch::Neon.rank() {
                            return unsafe { $neon(a, b, res) };
                        }
                    )?
                }

                $(
                    if arch.rank() >= $crate::cpu::dispatch::Arch::Swar.rank() {
                        return $swar(a, b, res);
                    }
                )?

                $scalar(a, b, res)
            }

            let n = a.len();
            assert_eq!(n, b.len(), concat!(stringify!($name), ": lhs/rhs length mismatch"));
            assert_eq!(n, res.len(), concat!(stringify!($name), ": output length mismatch"));

            // ---- Tier III: rayon gate --------------------------------------
            let threshold = $crate::cpu::thresholds::get($threshold);
            if n < threshold {
                __serial(a, b, res);
                return;
            }

            {
                use rayon::prelude::*;
                // `par_chunks_mut` yields a short final chunk when `n % threshold
                // != 0`; slicing `a`/`b` by `chunk.len()` handles that tail with
                // no separate code path.
                let chunk_len = threshold.max(1);
                res.par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        let start = i * chunk_len;
                        let end = start + chunk.len();
                        __serial(&a[start..end], &b[start..end], chunk);
                    });
            }
        }
    };
}

/// Generate the Tier II/III scaffolding for an **elementwise unary** kernel.
///
/// Two forms:
///
/// * out-of-place — kernels have signature `f(&[T], &mut [T])`:
///   ```ignore
///   elementwise_unary! {
///       name: relu_f32,
///       elem: f32,
///       threshold: crate::cpu::thresholds::Threshold::GeluF32,
///       scalar: path::to::relu_f32_scalar,
///       avx1:   path::to::relu_f32_avx1,
///   }
///   ```
/// * in-place — kernels have signature `f(&mut [T])`, selected by a leading
///   `inplace,` token:
///   ```ignore
///   elementwise_unary! {
///       inplace,
///       name: gelu_f32,
///       elem: f32,
///       threshold: crate::cpu::thresholds::Threshold::GeluF32,
///       scalar: path::to::gelu_f32_scalar,
///       avx2:   path::to::gelu_f32_avx2,
///   }
///   ```
///
/// The same field ordering and safety contract as [`elementwise_binary!`] applies.
#[macro_export]
macro_rules! elementwise_unary {
    // ---- in-place: f(&mut [T]) -----------------------------------------------
    (
        inplace,
        name: $name:ident,
        elem: $elem:ty,
        threshold: $threshold:expr,
        scalar: $scalar:path,
        $( swar: $swar:path, )?
        $( sse2: $sse2:path, )?
        $( avx1: $avx1:path, )?
        $( avx2: $avx2:path, )?
        $( neon: $neon:path, )?
    ) => {
        pub fn $name(buf: &mut [$elem]) {
            #[inline]
            fn __serial(buf: &mut [$elem]) {
                #[allow(unused_variables)]
                let arch = $crate::cpu::dispatch::active_arch();

                #[cfg(target_arch = "x86_64")]
                {
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Avx2.rank() {
                        return unsafe { $avx2(buf) };
                    } )?
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Avx1.rank() {
                        return unsafe { $avx1(buf) };
                    } )?
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Sse2.rank() {
                        return unsafe { $sse2(buf) };
                    } )?
                }

                #[cfg(target_arch = "aarch64")]
                {
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Neon.rank() {
                        return unsafe { $neon(buf) };
                    } )?
                }

                $( if arch.rank() >= $crate::cpu::dispatch::Arch::Swar.rank() {
                    return $swar(buf);
                } )?

                $scalar(buf)
            }

            let n = buf.len();
            let threshold = $crate::cpu::thresholds::get($threshold);
            if n < threshold {
                __serial(buf);
                return;
            }

            {
                use rayon::prelude::*;
                let chunk_len = threshold.max(1);
                buf.par_chunks_mut(chunk_len).for_each(|chunk| __serial(chunk));
            }
        }
    };

    // ---- out-of-place: f(&[T], &mut [T]) -------------------------------------
    (
        name: $name:ident,
        elem: $elem:ty,
        threshold: $threshold:expr,
        scalar: $scalar:path,
        $( swar: $swar:path, )?
        $( sse2: $sse2:path, )?
        $( avx1: $avx1:path, )?
        $( avx2: $avx2:path, )?
        $( neon: $neon:path, )?
    ) => {
        pub fn $name(src: &[$elem], res: &mut [$elem]) {
            #[inline]
            fn __serial(src: &[$elem], res: &mut [$elem]) {
                #[allow(unused_variables)]
                let arch = $crate::cpu::dispatch::active_arch();

                #[cfg(target_arch = "x86_64")]
                {
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Avx2.rank() {
                        return unsafe { $avx2(src, res) };
                    } )?
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Avx1.rank() {
                        return unsafe { $avx1(src, res) };
                    } )?
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Sse2.rank() {
                        return unsafe { $sse2(src, res) };
                    } )?
                }

                #[cfg(target_arch = "aarch64")]
                {
                    $( if arch.rank() >= $crate::cpu::dispatch::Arch::Neon.rank() {
                        return unsafe { $neon(src, res) };
                    } )?
                }

                $( if arch.rank() >= $crate::cpu::dispatch::Arch::Swar.rank() {
                    return $swar(src, res);
                } )?

                $scalar(src, res)
            }

            let n = src.len();
            assert_eq!(n, res.len(), concat!(stringify!($name), ": output length mismatch"));

            let threshold = $crate::cpu::thresholds::get($threshold);
            if n < threshold {
                __serial(src, res);
                return;
            }

            {
                use rayon::prelude::*;
                let chunk_len = threshold.max(1);
                res.par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        let start = i * chunk_len;
                        let end = start + chunk.len();
                        __serial(&src[start..end], chunk);
                    });
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arch override is process-global, so every test that touches it must
    /// hold this lock or they will race each other under the default
    /// multi-threaded test runner.
    pub(super) static ARCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Poison-tolerant: a panicking test must not disable every other test.
    pub(super) fn arch_guard() -> std::sync::MutexGuard<'static, ()> {
        ARCH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn detect_is_stable_and_family_correct() {
        let d = detect_arch();
        assert!(d.family_matches_target(), "detected {:?} off-family", d);
        assert_eq!(d, detect_arch());
    }

    #[test]
    fn override_round_trips_and_clamps() {
        let _g = arch_guard();
        with_forced_arch(None, || {
            let detected = detect_arch();

            // Forcing down always works.
            force_arch(Some(Arch::Scalar));
            assert_eq!(active_arch(), Arch::Scalar);

            // Forcing an off-family arch is ignored.
            let off_family = if cfg!(target_arch = "x86_64") {
                Arch::Neon
            } else {
                Arch::Avx2
            };
            force_arch(Some(off_family));
            assert_eq!(active_arch(), detected);

            // Clearing restores detection.
            force_arch(None);
            assert_eq!(active_arch(), detected);
        });
    }

    /// Expands both macros with stand-in kernels. The variants deliberately
    /// compute *different* functions, so the assertions prove which rung of the
    /// ladder actually ran — not merely that the code compiled.
    mod macro_expansion {
        use crate::cpu::dispatch::{self, Arch};
        use crate::cpu::thresholds::{self, Threshold};

        pub fn marker_scalar(a: &[f32], b: &[f32], r: &mut [f32]) {
            for i in 0..a.len() {
                r[i] = a[i] - b[i];
            }
        }
        /// Stands in for a vector kernel: same `unsafe fn` shape the real ones
        /// have, but computes `a + b` so it is distinguishable from scalar.
        pub unsafe fn marker_vector(a: &[f32], b: &[f32], r: &mut [f32]) {
            for i in 0..a.len() {
                r[i] = a[i] + b[i];
            }
        }

        pub fn unary_scalar(src: &[f32], r: &mut [f32]) {
            for i in 0..src.len() {
                r[i] = -src[i];
            }
        }
        pub fn inplace_scalar(buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v *= 2.0;
            }
        }

        crate::elementwise_binary! {
            name: probe_binary,
            elem: f32,
            threshold: Threshold::AddI8,
            scalar: marker_scalar,
            sse2: marker_vector,
        }

        crate::elementwise_binary! {
            name: probe_binary_empty_probe,
            elem: f32,
            threshold: Threshold::DivF16,
            scalar: marker_scalar,
        }

        crate::elementwise_unary! {
            name: probe_unary,
            elem: f32,
            threshold: Threshold::SubF16,
            scalar: unary_scalar,
        }

        crate::elementwise_unary! {
            inplace,
            name: probe_unary_inplace,
            elem: f32,
            threshold: Threshold::MulI8,
            scalar: inplace_scalar,
        }

        #[test]
        fn ladder_honours_forced_arch() {
            let _g = super::arch_guard();
            let a = vec![10.0f32; 8];
            let b = vec![3.0f32; 8];
            let mut out = vec![0.0f32; 8];

            dispatch::with_forced_arch(Some(Arch::Scalar), || {
                probe_binary(&a, &b, &mut out);
            });
            assert_eq!(out[0], 7.0, "forcing Scalar must run the scalar kernel");

            if cfg!(target_arch = "x86_64") && dispatch::detect_arch().rank() >= Arch::Sse2.rank() {
                dispatch::with_forced_arch(Some(Arch::Sse2), || {
                    probe_binary(&a, &b, &mut out);
                });
                assert_eq!(out[0], 13.0, "forcing Sse2 must run the vector kernel");
            }
        }

        /// The parallel path must produce exactly the serial result, including
        /// the short final chunk when `n % threshold != 0`.
        #[test]
        fn parallel_gate_and_tail_match_serial() {
            let _g = super::arch_guard();
            let n = 1000usize;
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

            let mut serial = vec![0.0f32; n];
            thresholds::set(Threshold::AddI8, usize::MAX - 1); // force serial
            dispatch::with_forced_arch(Some(Arch::Scalar), || probe_binary(&a, &b, &mut serial));

            let mut parallel = vec![0.0f32; n];
            thresholds::set(Threshold::AddI8, 64); // 1000 = 15*64 + 40 -> short tail
            dispatch::with_forced_arch(Some(Arch::Scalar), || probe_binary(&a, &b, &mut parallel));

            thresholds::reset(Threshold::AddI8);
            assert_eq!(serial, parallel);
            assert_eq!(parallel[999], 999.0 - 1998.0);
        }

        #[test]
        fn unary_forms_round_trip() {
            let src: Vec<f32> = (0..300).map(|i| i as f32).collect();
            let mut out = vec![0.0f32; 300];

            thresholds::set(Threshold::SubF16, 32);
            probe_unary(&src, &mut out);
            thresholds::reset(Threshold::SubF16);
            assert_eq!(out[299], -299.0);

            let mut buf = src.clone();
            thresholds::set(Threshold::MulI8, 32);
            probe_unary_inplace(&mut buf);
            thresholds::reset(Threshold::MulI8);
            assert_eq!(buf[299], 598.0);
        }

        #[test]
        fn empty_input_is_not_a_panic() {
            let empty: [f32; 0] = [];
            let mut out: [f32; 0] = [];
            thresholds::set(Threshold::DivF16, 0);
            probe_binary_empty_probe(&empty, &empty, &mut out);
            thresholds::reset(Threshold::DivF16);
        }
    }

    #[test]
    fn names_round_trip() {
        for a in [
            Arch::Scalar,
            Arch::Swar,
            Arch::Sse2,
            Arch::Avx1,
            Arch::Avx2,
            Arch::Avx512,
            Arch::Neon,
        ] {
            assert_eq!(Arch::from_name(a.name()), Some(a));
        }
        assert_eq!(Arch::from_name("avx512"), Some(Arch::Avx512));
        assert_eq!(Arch::from_name("bogus"), None);
    }
}
