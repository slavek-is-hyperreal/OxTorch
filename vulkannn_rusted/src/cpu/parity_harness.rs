//! parity_harness.rs — differential testing of migrated kernels against `cpu_old`.
//!
//! Test-only (`#[cfg(test)]`): nothing here ships in a release build.
//!
//! # The expiry problem
//! Comparing `src/cpu/` against `src/cpu_old/` is the strongest check available
//! *today*, but `cpu_old` is deleted in Wave 6 — at which point a pure
//! legacy-comparison test evaporates and takes its coverage with it.
//!
//! So every parity run does two things:
//!
//! * compares the new kernel against the legacy kernel (the branch with an expiry
//!   date), **and**
//! * compares the new kernel against a recorded snapshot in
//!   `tests/golden/{op}_{dtype}.bin`, when that file exists (the branch that
//!   survives).
//!
//! Regenerate snapshots with `OXTORCH_GOLDEN_REGEN=1 cargo test`. Wave 6 deletes
//! only the legacy branch — [`run_binary_parity`]'s `legacy_fn` argument and its
//! call site — and the golden branch keeps working unchanged.
//!
//! **Regenerating is a deliberate act.** `OXTORCH_GOLDEN_REGEN=1` overwrites the
//! recorded outputs with whatever the code currently produces; run it only after
//! the legacy comparison passes, and review the resulting diff.
//!
//! # What is exercised
//! Independently of the shapes a caller passes, every run also covers
//! [`MANDATORY_LENS`] — lengths chosen so that `n % vector_width != 0` for widths
//! 4 (SSE2), 8 (AVX), 16 (AVX-512 / NEON x4 unroll), 32 and 64 (the unrolled
//! AVX1/AVX2 bodies in this repo) — and the input vectors always begin with the
//! full [`SPECIALS`] table: ±0, ±inf, NaN, and denormals.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

/// Element types the harness can drive.
///
/// `to_f32` is the canonical wire format for golden files: every dtype is stored
/// as f32 so the snapshot layout does not change when a new dtype is migrated.
pub trait ParityElem: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Short dtype tag used in golden file names (`sub_f32.bin`).
    const NAME: &'static str;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    /// Bit-exact comparison (distinguishes +0.0 from -0.0, and NaN payloads).
    fn bits_eq(self, other: Self) -> bool;
}

impl ParityElem for f32 {
    const NAME: &'static str = "f32";
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl ParityElem for half::bf16 {
    const NAME: &'static str = "bf16";
    #[inline]
    fn from_f32(v: f32) -> Self {
        half::bf16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }
    #[inline]
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl ParityElem for half::f16 {
    const NAME: &'static str = "f16";
    #[inline]
    fn from_f32(v: f32) -> Self {
        half::f16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
    #[inline]
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl ParityElem for i8 {
    const NAME: &'static str = "i8";
    #[inline]
    fn from_f32(v: f32) -> Self {
        if v.is_nan() {
            0
        } else {
            v.clamp(i8::MIN as f32, i8::MAX as f32) as i8
        }
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn bits_eq(self, other: Self) -> bool {
        self == other
    }
}

/// Map the dtype token used in `assert_parity_vs_legacy!` to a concrete type.
/// Extend here when a new dtype is migrated.
#[macro_export]
macro_rules! __parity_ty {
    (f32) => { f32 };
    (f16) => { half::f16 };
    (bf16) => { half::bf16 };
    (i8) => { i8 };
}

/// Values that must appear in every input vector long enough to hold them.
/// Ordered so that pairing `a[i]` with a rotated `b[i]` produces the nasty
/// combinations (inf - inf, 0/0, NaN ⊕ finite, denormal ⊕ denormal).
pub const SPECIALS: &[f32] = &[
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    -f32::NAN,
    f32::MIN_POSITIVE,   // smallest normal
    -f32::MIN_POSITIVE,
    1e-40,               // denormal
    -1e-40,
    1.401_298_5e-45,     // f32::from_bits(1), smallest denormal
    -1.401_298_5e-45,
    f32::MAX,
    f32::MIN,
    1.0,
    -1.0,
];

/// Lengths every parity run covers, on top of whatever the caller asks for.
/// Chosen to break every vector width and unroll factor used in this repo.
pub const MANDATORY_LENS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025,
];

/// Deterministic pseudo-random filler (xorshift32). Never `rand`: the harness must
/// produce byte-identical inputs on every host so golden files stay comparable.
fn fill_pseudo_random(out: &mut [f32], mut state: u32) {
    for slot in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Map to roughly [-8, 8) with a spread of magnitudes.
        let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
        *slot = (unit - 0.5) * 16.0;
    }
}

/// Build one input vector: specials first, deterministic noise after.
/// `rotate` shifts the specials table so the second operand pairs differently
/// from the first.
pub fn make_input<T: ParityElem>(n: usize, rotate: usize, seed: u32) -> Vec<T> {
    let mut raw = vec![0f32; n];
    let head = n.min(SPECIALS.len());
    for i in 0..head {
        raw[i] = SPECIALS[(i + rotate) % SPECIALS.len()];
    }
    if n > head {
        fill_pseudo_random(&mut raw[head..], seed | 1);
    }
    raw.into_iter().map(T::from_f32).collect()
}

/// Compare two result buffers.
///
/// `tol == 0.0` demands bit-exact equality (the right setting for add/sub/mul,
/// where SIMD and scalar must agree exactly). A positive `tol` is an absolute
/// tolerance on the f32 view, applied only to finite values; NaN must still meet
/// NaN and each infinity must meet the same-signed infinity.
fn compare<T: ParityElem>(ctx: &str, n: usize, got: &[T], want: &[T], tol: f64) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch at n={n}");
    for i in 0..got.len() {
        let g = got[i];
        let w = want[i];
        if g.bits_eq(w) {
            continue;
        }
        let gf = g.to_f32();
        let wf = w.to_f32();
        if gf.is_nan() && wf.is_nan() {
            // NaN payload/sign differences between SIMD and scalar are permitted;
            // NaN-ness itself is not.
            continue;
        }
        if tol == 0.0 {
            panic!(
                "{ctx}: n={n} idx={i} not bit-identical: got {gf:?} ({g:?}) want {wf:?} ({w:?})"
            );
        }
        if gf.is_nan() != wf.is_nan() {
            panic!("{ctx}: n={n} idx={i} NaN-ness differs: got {gf:?} want {wf:?}");
        }
        if gf.is_infinite() || wf.is_infinite() {
            panic!("{ctx}: n={n} idx={i} infinity mismatch: got {gf:?} want {wf:?}");
        }
        let diff = (gf as f64 - wf as f64).abs();
        assert!(
            diff <= tol,
            "{ctx}: n={n} idx={i} |{gf:?} - {wf:?}| = {diff} > tol {tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Golden snapshots
// ---------------------------------------------------------------------------

const GOLDEN_MAGIC: &[u8; 8] = b"OXGOLD01";
const GOLDEN_REGEN_ENV: &str = "OXTORCH_GOLDEN_REGEN";

/// One recorded case: the inputs and the output the kernel produced for them.
pub struct GoldenCase {
    pub n: usize,
    pub a: Vec<f32>,
    /// `None` for unary ops.
    pub b: Option<Vec<f32>>,
    pub out: Vec<f32>,
}

fn golden_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/vulkannn_rusted`; the golden files live in
    // the repo-level `tests/` tree next to the Python parity suite.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("golden")
}

fn golden_path(op: &str, dtype: &str) -> PathBuf {
    golden_dir().join(format!("{op}_{dtype}.bin"))
}

fn regen_requested() -> bool {
    matches!(
        std::env::var(GOLDEN_REGEN_ENV).ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Trailing path segment of a `stringify!`d path, e.g. `a :: b :: sub_f32` -> `sub_f32`.
pub fn op_name_of(path: &str) -> String {
    path.rsplit("::").next().unwrap_or(path).trim().to_string()
}

fn write_golden(op: &str, dtype: &str, cases: &[GoldenCase]) {
    let dir = golden_dir();
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    let path = golden_path(op, dtype);
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(GOLDEN_MAGIC);
    buf.push(if cases.first().map(|c| c.b.is_some()).unwrap_or(false) { 2 } else { 1 });
    buf.extend_from_slice(&(cases.len() as u32).to_le_bytes());
    for c in cases {
        buf.extend_from_slice(&(c.n as u64).to_le_bytes());
        for chunk in [Some(&c.a), c.b.as_ref(), Some(&c.out)].into_iter().flatten() {
            for v in chunk.iter() {
                buf.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    let mut f = fs::File::create(&path)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    f.write_all(&buf).expect("golden write failed");
    eprintln!("[parity] regenerated {} ({} cases)", path.display(), cases.len());
}

fn read_golden(op: &str, dtype: &str) -> Option<Vec<GoldenCase>> {
    let path = golden_path(op, dtype);
    let bytes = fs::read(&path).ok()?;
    assert!(
        bytes.len() >= 13 && &bytes[..8] == GOLDEN_MAGIC,
        "{}: not an OxTorch golden file (regenerate with {GOLDEN_REGEN_ENV}=1)",
        path.display()
    );
    let arity = bytes[8] as usize;
    let count = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
    let mut off = 13;
    let mut cases = Vec::with_capacity(count);
    let mut take = |off: &mut usize, n: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
            v.push(f32::from_bits(raw));
            *off += 4;
        }
        v
    };
    for _ in 0..count {
        let n = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let a = take(&mut off, n);
        let b = if arity == 2 { Some(take(&mut off, n)) } else { None };
        let out = take(&mut off, n);
        cases.push(GoldenCase { n, a, b, out });
    }
    Some(cases)
}

/// Check freshly produced cases against a recorded snapshot, if one exists.
/// Missing snapshot is not a failure — it just means this op has no baseline yet.
fn check_golden(op: &str, dtype: &str, cases: &[GoldenCase], tol: f64) {
    // The harness's own self-tests must never create or consult snapshot files.
    if op.starts_with("smoke_") || op.starts_with("self_test_") {
        return;
    }
    if regen_requested() {
        write_golden(op, dtype, cases);
        return;
    }
    let Some(recorded) = read_golden(op, dtype) else {
        eprintln!(
            "[parity] no golden snapshot for {op}_{dtype}; run {GOLDEN_REGEN_ENV}=1 cargo test to create one"
        );
        return;
    };
    assert_eq!(
        recorded.len(),
        cases.len(),
        "{op}_{dtype}: golden has {} cases, run produced {} — regenerate with {GOLDEN_REGEN_ENV}=1",
        recorded.len(),
        cases.len()
    );
    for (rec, cur) in recorded.iter().zip(cases.iter()) {
        assert_eq!(rec.n, cur.n, "{op}_{dtype}: golden case length drifted");
        compare_f32(&format!("{op}_{dtype} golden inputs(a)"), rec.n, &cur.a, &rec.a, 0.0);
        if let (Some(rb), Some(cb)) = (rec.b.as_ref(), cur.b.as_ref()) {
            compare_f32(&format!("{op}_{dtype} golden inputs(b)"), rec.n, cb, rb, 0.0);
        }
        compare_f32(&format!("{op}_{dtype} golden output"), rec.n, &cur.out, &rec.out, tol);
    }
}

fn compare_f32(ctx: &str, n: usize, got: &[f32], want: &[f32], tol: f64) {
    compare::<f32>(ctx, n, got, want, tol);
}

fn shape_plan(extra: &[usize]) -> Vec<usize> {
    let mut lens: Vec<usize> = MANDATORY_LENS.to_vec();
    lens.extend_from_slice(extra);
    lens.sort_unstable();
    lens.dedup();
    lens
}

/// Drive a binary kernel against its legacy counterpart and against the snapshot.
///
/// Wave 6 removes `legacy_fn` and the block marked `LEGACY BRANCH`; everything
/// else stays.
pub fn run_binary_parity<T: ParityElem>(
    new_path: &str,
    dtype: &str,
    new_fn: &dyn Fn(&[T], &[T], &mut [T]),
    legacy_fn: Option<&dyn Fn(&[T], &[T], &mut [T])>,
    extra_shapes: &[usize],
    tol: f64,
) {
    let op = op_name_of(new_path);
    let mut cases = Vec::new();

    for &n in shape_plan(extra_shapes).iter() {
        let a: Vec<T> = make_input(n, 0, 0x9E37_79B9);
        let b: Vec<T> = make_input(n, 3, 0x85EB_CA6B);

        let mut got = vec![T::from_f32(0.0); n];
        new_fn(&a, &b, &mut got);

        // ---- LEGACY BRANCH (deleted in Wave 6) ----------------------------
        if let Some(legacy) = legacy_fn {
            let mut want = vec![T::from_f32(0.0); n];
            legacy(&a, &b, &mut want);
            compare(&format!("{op}_{dtype} vs legacy"), n, &got, &want, tol);
        }
        // ---- END LEGACY BRANCH --------------------------------------------

        cases.push(GoldenCase {
            n,
            a: a.iter().map(|v| v.to_f32()).collect(),
            b: Some(b.iter().map(|v| v.to_f32()).collect()),
            out: got.iter().map(|v| v.to_f32()).collect(),
        });
    }

    check_golden(&op, dtype, &cases, tol);
}

/// Unary counterpart of [`run_binary_parity`], for kernels shaped
/// `f(&[T], &mut [T])`.
pub fn run_unary_parity<T: ParityElem>(
    new_path: &str,
    dtype: &str,
    new_fn: &dyn Fn(&[T], &mut [T]),
    legacy_fn: Option<&dyn Fn(&[T], &mut [T])>,
    extra_shapes: &[usize],
    tol: f64,
) {
    let op = op_name_of(new_path);
    let mut cases = Vec::new();

    for &n in shape_plan(extra_shapes).iter() {
        let a: Vec<T> = make_input(n, 0, 0x9E37_79B9);

        let mut got = vec![T::from_f32(0.0); n];
        new_fn(&a, &mut got);

        // ---- LEGACY BRANCH (deleted in Wave 6) ----------------------------
        if let Some(legacy) = legacy_fn {
            let mut want = vec![T::from_f32(0.0); n];
            legacy(&a, &mut want);
            compare(&format!("{op}_{dtype} vs legacy"), n, &got, &want, tol);
        }
        // ---- END LEGACY BRANCH --------------------------------------------

        cases.push(GoldenCase {
            n,
            a: a.iter().map(|v| v.to_f32()).collect(),
            b: None,
            out: got.iter().map(|v| v.to_f32()).collect(),
        });
    }

    check_golden(&op, dtype, &cases, tol);
}

/// Generate a `#[test]` that runs a migrated kernel and its legacy counterpart on
/// identical data, then checks the result against `tests/golden/{op}_{dtype}.bin`.
///
/// ```ignore
/// mod parity {
///     use super::*;
///     assert_parity_vs_legacy!(
///         crate::cpu::ops::binary::sub::sub_f32,
///         crate::cpu_old::ops::binary::sub::sub_f32,
///         f32,
///         [4096, 100_000],
///         0.0
///     );
/// }
/// ```
///
/// Arity and forms:
/// * `(new_fn, legacy_fn, dtype, [shapes], tol)` — binary op, generates
///   `fn parity_vs_legacy()`. One invocation per enclosing module.
/// * `(test_name, new_fn, legacy_fn, dtype, [shapes], tol)` — same, with an
///   explicit test-function name so several can share a module.
/// * `(unary, ...)` / `(unary, test_name, ...)` — the `f(&[T], &mut [T])` forms.
///
/// `dtype` is one of `f32`, `f16`, `bf16`, `i8` (see `__parity_ty!`). `shapes` is
/// a bracketed list of extra lengths; [`MANDATORY_LENS`] is always added on top.
/// `tol` of `0.0` demands bit-exact agreement.
#[macro_export]
macro_rules! assert_parity_vs_legacy {
    (unary, $test_name:ident, $new:path, $legacy:path, $dtype:ident, [$($n:expr),* $(,)?], $tol:expr) => {
        #[test]
        fn $test_name() {
            let shapes: &[usize] = &[$($n),*];
            $crate::cpu::parity_harness::run_unary_parity::<$crate::__parity_ty!($dtype)>(
                stringify!($new),
                stringify!($dtype),
                &|src, res| $new(src, res),
                Some(&|src, res| $legacy(src, res)),
                shapes,
                $tol,
            );
        }
    };
    (unary, $new:path, $legacy:path, $dtype:ident, [$($n:expr),* $(,)?], $tol:expr) => {
        $crate::assert_parity_vs_legacy!(unary, parity_vs_legacy, $new, $legacy, $dtype, [$($n),*], $tol);
    };
    ($test_name:ident, $new:path, $legacy:path, $dtype:ident, [$($n:expr),* $(,)?], $tol:expr) => {
        #[test]
        fn $test_name() {
            let shapes: &[usize] = &[$($n),*];
            $crate::cpu::parity_harness::run_binary_parity::<$crate::__parity_ty!($dtype)>(
                stringify!($new),
                stringify!($dtype),
                &|a, b, res| $new(a, b, res),
                Some(&|a, b, res| $legacy(a, b, res)),
                shapes,
                $tol,
            );
        }
    };
    ($new:path, $legacy:path, $dtype:ident, [$($n:expr),* $(,)?], $tol:expr) => {
        $crate::assert_parity_vs_legacy!(parity_vs_legacy, $new, $legacy, $dtype, [$($n),*], $tol);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_are_present_in_generated_inputs() {
        let v: Vec<f32> = make_input(64, 0, 1);
        assert_eq!(v[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(v[1].to_bits(), (-0.0f32).to_bits());
        assert!(v[2].is_infinite() && v[2] > 0.0);
        assert!(v[3].is_infinite() && v[3] < 0.0);
        assert!(v[4].is_nan());
        assert!(v[8] != 0.0 && v[8].is_subnormal(), "expected denormal, got {}", v[8]);
    }

    #[test]
    fn generation_is_deterministic() {
        let a: Vec<f32> = make_input(1025, 0, 7);
        let b: Vec<f32> = make_input(1025, 0, 7);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn operands_are_rotated_against_each_other() {
        let a: Vec<f32> = make_input(16, 0, 1);
        let b: Vec<f32> = make_input(16, 3, 2);
        assert_ne!(a[0].to_bits(), b[0].to_bits());
    }

    #[test]
    fn shape_plan_covers_mandatory_and_extra() {
        let plan = shape_plan(&[4096]);
        for &m in MANDATORY_LENS {
            assert!(plan.contains(&m), "missing mandatory len {m}");
        }
        assert!(plan.contains(&4096));
        assert!(plan.windows(2).all(|w| w[0] < w[1]), "plan must be sorted+deduped");
    }

    #[test]
    fn op_name_extraction() {
        assert_eq!(op_name_of("crate :: cpu :: ops :: sub_f32"), "sub_f32");
        assert_eq!(op_name_of("sub_f32"), "sub_f32");
    }

    #[test]
    fn bit_exact_comparison_rejects_signed_zero_drift() {
        let got = [0.0f32];
        let want = [-0.0f32];
        let r = std::panic::catch_unwind(|| compare::<f32>("t", 1, &got, &want, 0.0));
        assert!(r.is_err(), "+0.0 vs -0.0 must fail at tol = 0");
    }

    #[test]
    fn nan_meets_nan() {
        let got = [f32::NAN];
        let want = [-f32::NAN];
        compare::<f32>("t", 1, &got, &want, 0.0);
    }

    /// Compile-and-run smoke test for `assert_parity_vs_legacy!` itself, so a
    /// syntax break in the macro is caught in Wave 0 rather than by the first
    /// worker who tries to use it.
    mod macro_smoke {
        pub fn smoke_sub_f32_new(a: &[f32], b: &[f32], r: &mut [f32]) {
            for i in 0..a.len() {
                r[i] = a[i] - b[i];
            }
        }
        pub fn smoke_sub_f32_legacy(a: &[f32], b: &[f32], r: &mut [f32]) {
            for i in 0..a.len() {
                r[i] = a[i] - b[i];
            }
        }
        pub fn smoke_neg_f32_new(src: &[f32], r: &mut [f32]) {
            for i in 0..src.len() {
                r[i] = -src[i];
            }
        }
        pub fn smoke_neg_f32_legacy(src: &[f32], r: &mut [f32]) {
            for i in 0..src.len() {
                r[i] = -src[i];
            }
        }

        crate::assert_parity_vs_legacy!(
            binary_form,
            smoke_sub_f32_new,
            smoke_sub_f32_legacy,
            f32,
            [4096],
            0.0
        );

        crate::assert_parity_vs_legacy!(
            unary,
            unary_form,
            smoke_neg_f32_new,
            smoke_neg_f32_legacy,
            f32,
            [4096],
            0.0
        );
    }

    /// Self-test of the whole pipeline: a deliberately wrong "new" kernel must be
    /// caught by the legacy comparison.
    #[test]
    fn harness_detects_a_broken_kernel() {
        fn good(a: &[f32], b: &[f32], r: &mut [f32]) {
            for i in 0..a.len() {
                r[i] = a[i] - b[i];
            }
        }
        fn broken(a: &[f32], b: &[f32], r: &mut [f32]) {
            for i in 0..a.len() {
                // Wrong only on the tail, which is exactly what a bad SIMD
                // remainder loop looks like.
                r[i] = if i >= 8 && i % 8 != 0 { a[i] + b[i] } else { a[i] - b[i] };
            }
        }
        let r = std::panic::catch_unwind(|| {
            run_binary_parity::<f32>(
                "self_test_sub",
                "f32",
                &|a, b, o| broken(a, b, o),
                Some(&|a, b, o| good(a, b, o)),
                &[],
                0.0,
            )
        });
        assert!(r.is_err(), "harness failed to catch a broken tail loop");
    }
}
