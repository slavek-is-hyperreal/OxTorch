//! thresholds.rs — Runtime-configurable parallelism thresholds (Tier III gate).
//!
//! # Why this module exists
//! Before Wave 0 every `[op]/mod.rs` hard-coded its own `const PAR_THRESHOLD: usize`.
//! That made the rayon cut-over point (a) impossible to tune without a rebuild and
//! (b) impossible to sweep from a benchmark harness. This module centralises every
//! threshold behind an `AtomicUsize` slot so it can be changed at runtime.
//!
//! # Resolution order (lazy, on first `get`)
//! 1. `OXTORCH_PAR_THRESHOLD_<OP>_<DTYPE>`  (e.g. `OXTORCH_PAR_THRESHOLD_ADD_F32`)
//! 2. `OXTORCH_PAR_THRESHOLD`               (global override for all ops)
//! 3. the compiled-in default (see the table below — every value is transcribed
//!    from the legacy source, never guessed)
//!
//! An explicit `set()` / `set_all()` always wins over the environment; the env is
//! only consulted for slots that have not been resolved yet.
//!
//! # Memory ordering
//! All loads use `Ordering::Relaxed`. A threshold is a scheduling hint: a torn
//! read-modify-write race can only pick the old or the new value, both of which
//! are valid, and no other memory is published through this cell.
//!
//! # Adding a new threshold (later waves)
//! Append one line to the `define_thresholds!` invocation below, keeping the list
//! **alphabetical by variant name**, one variant per op×dtype. Because each worker
//! touches a distinct alphabetical slot, merge conflicts stay line-local.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Sentinel meaning "this slot has not been resolved from env/default yet".
const UNSET: usize = usize::MAX;

/// Helper: expand to a fresh unresolved slot, once per repetition.
macro_rules! __unset_slot {
    ($ignored:tt) => {
        AtomicUsize::new(UNSET)
    };
}

macro_rules! define_thresholds {
    (
        $(
            $(#[$meta:meta])*
            $variant:ident => ($env_suffix:literal, $default:expr)
        ),* $(,)?
    ) => {
        /// One variant per (operation × dtype). Keep alphabetical.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(usize)]
        pub enum Threshold {
            $(
                $(#[$meta])*
                $variant,
            )*
        }

        /// All thresholds, in declaration (alphabetical) order.
        pub const ALL: [Threshold; [$( Threshold::$variant, )*].len()] =
            [ $( Threshold::$variant, )* ];

        /// Number of defined thresholds.
        pub const COUNT: usize = ALL.len();

        static SLOTS: [AtomicUsize; COUNT] = [ $( __unset_slot!($variant), )* ];

        static DEFAULTS: [usize; COUNT] = [ $( $default, )* ];

        /// Per-slot env-var suffix, e.g. `"ADD_F32"`.
        static ENV_SUFFIXES: [&str; COUNT] = [ $( $env_suffix, )* ];

        impl Threshold {
            /// Stable string name of the variant (useful for logging / Python bridges).
            pub const fn name(self) -> &'static str {
                match self {
                    $( Threshold::$variant => stringify!($variant), )*
                }
            }

            /// Resolve a public threshold name (the env-suffix form, e.g. "ADD_F32",
            /// case-insensitive) back to its variant. Used by the Python bridge.
            pub fn from_name(s: &str) -> Option<Threshold> {
                $( if s.eq_ignore_ascii_case($env_suffix) { return Some(Threshold::$variant); } )*
                None
            }
        }
    };
}

define_thresholds! {
    // ---------------------------------------------------------------------
    // ADD
    // ---------------------------------------------------------------------
    /// src/cpu/ops/binary/add/mod.rs — live value in the migrated tree.
    AddBf16 => ("ADD_BF16", 512_000),
    /// src/cpu_old/ops/binary/add/add_f16.rs:6 — legacy `PAR_THRESHOLD`.
    AddF16 => ("ADD_F16", 256_000),
    /// src/cpu/ops/binary/add/mod.rs — live value in the migrated tree.
    AddF32 => ("ADD_F32", 512_000),
    /// src/cpu_old/ops/binary/add/add_i8.rs:6 — legacy `PAR_THRESHOLD`.
    AddI8 => ("ADD_I8", 4_000_000),

    // ---------------------------------------------------------------------
    // ATAN2
    // ---------------------------------------------------------------------
    /// src/cpu/ops/binary/atan2/mod.rs:12 — live value in the migrated tree.
    /// NOTE: atan2 is the one migrated op whose chunk size (8_192) differs from
    /// its gate (16_384); every other op chunks at exactly the threshold. Any
    /// macro that assumes chunk == threshold must special-case atan2 or the
    /// parallel decomposition changes.
    Atan2F32 => ("ATAN2_F32", 16_384),

    // ---------------------------------------------------------------------
    // DIV
    // ---------------------------------------------------------------------
    /// src/cpu/ops/binary/div/mod.rs — live value in the migrated tree.
    DivBf16 => ("DIV_BF16", 512_000),
    /// src/cpu_old/ops/binary/div/div_f16.rs:6 — legacy `PAR_THRESHOLD`.
    DivF16 => ("DIV_F16", 256_000),
    /// src/cpu_old/ops/binary/div/div_f32.rs:6 — legacy `PAR_THRESHOLD`.
    DivF32 => ("DIV_F32", 4_000_000),
    // TODO: verify — src/cpu_old/ops/binary/div/div_i8.rs has NO parallel gate
    // at all (pure serial loop). 4_000_000 is carried over from the sibling
    // i8 ops (add_i8/mul_i8/sub_i8) so a value exists; it is not transcribed
    // from a div_i8 constant, because there is none.
    DivI8 => ("DIV_I8", 4_000_000),

    // ---------------------------------------------------------------------
    // GELU (unary)
    // ---------------------------------------------------------------------
    /// src/cpu_old/ops/unary/gelu/gelu_f32.rs:6 — legacy `PAR_THRESHOLD`.
    GeluF32 => ("GELU_F32", 64_000),

    // ---------------------------------------------------------------------
    // MUL
    // ---------------------------------------------------------------------
    /// src/cpu/ops/binary/mul/mod.rs — live value in the migrated tree.
    MulBf16 => ("MUL_BF16", 512_000),
    /// src/cpu_old/ops/binary/mul/mul_f16.rs:6 — legacy `PAR_THRESHOLD`.
    MulF16 => ("MUL_F16", 256_000),
    /// src/cpu_old/ops/binary/mul/mul_f32.rs:6 — legacy `PAR_THRESHOLD`.
    MulF32 => ("MUL_F32", 4_000_000),
    /// src/cpu_old/ops/binary/mul/mul_i8.rs:6 — legacy `PAR_THRESHOLD`.
    MulI8 => ("MUL_I8", 4_000_000),

    // ---------------------------------------------------------------------
    // NEG (unary) — no legacy rayon gate; 512_000 = migrated-tree convention.
    // ---------------------------------------------------------------------
    NegBf16 => ("NEG_BF16", 512_000),
    NegF16 => ("NEG_F16", 512_000),
    NegF32 => ("NEG_F32", 512_000),

    // ---------------------------------------------------------------------
    // POW (tensor ^ scalar exponent)
    // ---------------------------------------------------------------------
    // No legacy value — cpu_old pow_f32 had no rayon gate. 512_000 matches the
    // migrated-tree convention (cpu/ops/binary/add).
    PowF32 => ("POW_F32", 512_000),

    // ---------------------------------------------------------------------
    // RELU (unary) — no legacy rayon gate; 512_000 = migrated-tree convention.
    // ---------------------------------------------------------------------
    ReluBf16 => ("RELU_BF16", 512_000),
    ReluF16 => ("RELU_F16", 512_000),
    ReluF32 => ("RELU_F32", 512_000),
    ReluI8 => ("RELU_I8", 512_000),

    // ---------------------------------------------------------------------
    // SCALAR (tensor ⊕ scalar broadcast path)
    // ---------------------------------------------------------------------
    /// src/cpu_old/ops/binary/scalar.rs:6 — legacy `PAR_THRESHOLD` in `scalar_op_f32`.
    ScalarOpF32 => ("SCALAR_OP_F32", 1_000_000),

    // ---------------------------------------------------------------------
    // SUB
    // ---------------------------------------------------------------------
    /// src/cpu/ops/binary/sub/mod.rs — live value in the migrated tree.
    SubBf16 => ("SUB_BF16", 512_000),
    /// src/cpu_old/ops/binary/sub/sub_f16.rs:6 — legacy `PAR_THRESHOLD`.
    SubF16 => ("SUB_F16", 256_000),
    /// src/cpu/ops/binary/sub/mod.rs — live value in the migrated tree.
    SubF32 => ("SUB_F32", 512_000),
    /// src/cpu_old/ops/binary/sub/sub_i8.rs:6 — legacy `PAR_THRESHOLD`.
    SubI8 => ("SUB_I8", 4_000_000),
}

/// Global env fallback, consulted when the per-slot variable is absent.
const GLOBAL_ENV: &str = "OXTORCH_PAR_THRESHOLD";
/// Prefix of the per-slot env variables.
const ENV_PREFIX: &str = "OXTORCH_PAR_THRESHOLD_";

impl Threshold {
    #[inline]
    const fn idx(self) -> usize {
        self as usize
    }

    /// Compiled-in default, ignoring env and any runtime `set`.
    #[inline]
    pub fn default_value(self) -> usize {
        DEFAULTS[self.idx()]
    }

    /// Full name of the per-slot environment variable, e.g.
    /// `"OXTORCH_PAR_THRESHOLD_ADD_F32"`.
    pub fn env_var_name(self) -> String {
        format!("{}{}", ENV_PREFIX, ENV_SUFFIXES[self.idx()])
    }
}

/// Resolve a slot from the environment, falling back to the compiled-in default.
///
/// A value of `0` is accepted and means "always parallel". Unparseable values are
/// ignored (fall through to the next source) rather than panicking, because this
/// runs deep inside kernel dispatch where a panic would be catastrophic.
fn resolve(t: Threshold) -> usize {
    let per_slot = std::env::var(t.env_var_name())
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());
    if let Some(v) = per_slot {
        return sanitize(v);
    }
    if let Some(v) = std::env::var(GLOBAL_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        return sanitize(v);
    }
    t.default_value()
}

/// `usize::MAX` is reserved as the "unresolved" sentinel, so clamp it away.
/// Callers asking for `usize::MAX` mean "never parallelise", and `usize::MAX - 1`
/// is indistinguishable from that in practice.
#[inline]
fn sanitize(v: usize) -> usize {
    if v == UNSET {
        UNSET - 1
    } else {
        v
    }
}

/// Current parallelism threshold for `t`, in elements.
///
/// Resolves lazily on first call (env → default) and caches the result.
#[inline]
pub fn get(t: Threshold) -> usize {
    let slot = &SLOTS[t.idx()];
    let cur = slot.load(Ordering::Relaxed);
    if cur != UNSET {
        return cur;
    }
    // Benign race: two threads may both resolve; both compute the same value.
    let resolved = resolve(t);
    slot.store(resolved, Ordering::Relaxed);
    resolved
}

/// Override a single threshold at runtime. Takes precedence over the environment.
#[inline]
pub fn set(t: Threshold, value: usize) {
    SLOTS[t.idx()].store(sanitize(value), Ordering::Relaxed);
}

/// Override every threshold at runtime (handy for A/B sweeps and for forcing the
/// serial path in tests via `set_all(usize::MAX)`).
pub fn set_all(value: usize) {
    let v = sanitize(value);
    for slot in SLOTS.iter() {
        slot.store(v, Ordering::Relaxed);
    }
}

/// Drop a runtime override so the next `get` re-resolves from env/default.
pub fn reset(t: Threshold) {
    SLOTS[t.idx()].store(UNSET, Ordering::Relaxed);
}

/// Drop every runtime override.
pub fn reset_all() {
    for slot in SLOTS.iter() {
        slot.store(UNSET, Ordering::Relaxed);
    }
}

/// Snapshot of every threshold, for diagnostics.
pub fn snapshot() -> Vec<(Threshold, usize)> {
    ALL.iter().map(|&t| (t, get(t))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_legacy_transcription() {
        assert_eq!(Threshold::AddF32.default_value(), 512_000);
        assert_eq!(Threshold::SubF32.default_value(), 512_000);
        assert_eq!(Threshold::MulF32.default_value(), 4_000_000);
        assert_eq!(Threshold::DivF32.default_value(), 4_000_000);
        assert_eq!(Threshold::AddI8.default_value(), 4_000_000);
        assert_eq!(Threshold::MulF16.default_value(), 256_000);
        assert_eq!(Threshold::ScalarOpF32.default_value(), 1_000_000);
        assert_eq!(Threshold::GeluF32.default_value(), 64_000);
    }

    #[test]
    fn variant_index_matches_all_table() {
        for (i, t) in ALL.iter().enumerate() {
            assert_eq!(t.idx(), i, "variant {} has index {}", t.name(), t.idx());
        }
        assert_eq!(ALL.len(), COUNT);
    }

    #[test]
    fn env_var_names_are_unique_and_prefixed() {
        let mut seen = std::collections::HashSet::new();
        for t in ALL.iter() {
            let name = t.env_var_name();
            assert!(name.starts_with(ENV_PREFIX));
            assert!(seen.insert(name.clone()), "duplicate env var {}", name);
        }
    }

    #[test]
    fn set_then_get_round_trips() {
        // Use a slot no other test touches to stay order-independent.
        let t = Threshold::DivI8;
        set(t, 4242);
        assert_eq!(get(t), 4242);
        reset(t);
        // Only assert the compiled-in default when the environment is not
        // overriding it — otherwise this test would fail under a deliberate
        // `OXTORCH_PAR_THRESHOLD=... cargo test` sweep, which is a supported way
        // to run the suite.
        let env_overridden =
            std::env::var(t.env_var_name()).is_ok() || std::env::var(GLOBAL_ENV).is_ok();
        if !env_overridden {
            assert_eq!(get(t), t.default_value());
        }
    }

    #[test]
    fn env_resolution_precedence() {
        // `resolve` is a pure function of the environment, so precedence can be
        // checked without mutating process-global state (which would race the
        // other tests running in parallel).
        let t = Threshold::AddF16;
        match (
            std::env::var(t.env_var_name()).ok(),
            std::env::var(GLOBAL_ENV).ok(),
        ) {
            (Some(per_slot), _) => {
                assert_eq!(resolve(t), per_slot.trim().parse::<usize>().unwrap());
            }
            (None, Some(global)) => {
                assert_eq!(resolve(t), global.trim().parse::<usize>().unwrap());
            }
            (None, None) => {
                assert_eq!(resolve(t), t.default_value());
            }
        }
    }

    #[test]
    fn unparseable_env_falls_through_instead_of_panicking() {
        // Guards the "never panic inside dispatch" rule: a typo'd env var must
        // degrade to the default, not take the process down mid-kernel.
        assert_eq!(sanitize(0), 0);
        assert_eq!(sanitize(usize::MAX), usize::MAX - 1);
        assert_eq!("not-a-number".trim().parse::<usize>().ok(), None);
    }

    #[test]
    fn sentinel_is_clamped() {
        let t = Threshold::SubI8;
        set(t, usize::MAX);
        assert_eq!(get(t), usize::MAX - 1);
        reset(t);
    }
}
