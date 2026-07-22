# Known numerical divergences: OxTorch CPU vs PyTorch

This file collects every point where an OxTorch CPU kernel is known to produce a
different result from the PyTorch reference. It exists because the kernel
migration (`cpu_old` → `cpu`) is bound by **Rule 6** — *do not change numerics
during migration* — so divergences inherited from the legacy implementation are
transcribed faithfully rather than fixed in-flight. Each entry records a
**deliberate decision to defer**, to be reviewed in one pass after Wave 6.

Status legend: **KEEP** = documented behaviour, no change intended. **FIX** =
agreed to correct post-migration. **OPEN** = not yet decided.

---

## 1. `div(x, 0)` → `0.0` (torch: `±inf`) — OPEN

- **Where:** `cpu/ops/binary/div/*` (all dtypes). Scalar path guards `b==0` and
  returns `0.0`; the SIMD *body* divides raw (`±inf`/`NaN`) and only the scalar
  *tail* guards — so the result is even position-dependent within one call.
- **Legacy:** identical (`cpu_old/ops/binary/div`). Transcribed verbatim.
- **Severity:** medium. Masks division-by-zero that torch surfaces as inf.
- **Post-Wave-6 note:** decide between (a) keep the guard everywhere (consistent
  0.0), (b) drop the guard everywhere (match torch's inf), (c) leave as-is. The
  current SIMD-vs-scalar inconsistency is the one thing that should NOT survive
  review regardless of a/b/c.

## 2. `relu(NaN)` → `0.0` (torch: `NaN`) — OPEN, leaning FIX

- **Where:** `cpu/ops/unary/relu/*` (all dtypes). `x.max(0.0)` in Rust and
  `_mm*_max_ps`/`vmaxq_f32` all return the **non-NaN operand**, so a NaN input
  is silently zeroed.
- **Legacy:** identical. Transcribed verbatim (Rule 6).
- **Severity:** HIGH — higher than div/0. A NaN in activations is an alarm that
  propagates to the model output; torch lets the user see it, OxTorch silently
  extinguishes it. A model that visibly explodes with NaNs under PyTorch can
  produce quiet zeros under OxTorch and appear to "work". This directly
  undermines a drop-in-replacement claim.
- **Fix cost:** trivial — replace `max` with a compare-and-blend that keeps NaN
  (`x = (x > 0) ? x : 0` with NaN falling through), or an explicit `is_nan`
  select. Pennies of throughput on a memory-bound op for NaN-semantics parity.
- **Post-Wave-6 note:** recommended FIX. Left as KEEP only for the duration of
  the migration to honour Rule 6.

## 4. `neg(+0.0)` → `+0.0` on the SIMD path (torch/scalar: `-0.0`) — OPEN, leaning FIX

- **Where:** `cpu/ops/unary/neg/fp32/*` SIMD tiers compute `0 - x`
  (`_mm*_sub_ps(zero, v)` / not the NEON `vnegq`). IEEE `0 - (+0.0) = +0.0`, so
  the sign of zero is lost; the scalar path (`-x`) and torch both give `-0.0`.
  Yet another legacy SIMD-vs-scalar inconsistency (cf. div/0).
- **Legacy:** identical (`neg_f32_avx` uses `0 - x`). Transcribed verbatim.
- **Severity:** low (sign of zero rarely matters numerically) but breaks
  bit-exact drop-in on `+0.0` inputs.
- **Fix cost:** trivial and free — negate with a sign-bit XOR
  (`_mm*_xor_ps(v, set1(-0.0))`), the canonical negation, which yields `-0.0`
  correctly. NEON already uses `vnegq_f32` (correct).
- **Post-Wave-6 note:** recommended FIX (switch SIMD tiers to sign XOR).

## 3. `exp` (and dependents) — SIMD polynomial vs scalar `std::exp` — OPEN

- **Where:** `cpu/ops/unary/exp` and everything built on it (sigmoid, silu,
  tanh, gelu). Legacy computed the SIMD lanes with an ad-hoc low-degree
  polynomial while the scalar tail/fallback called `f32::exp()` (near-correctly
  rounded) — so accuracy depends on which lane an element landed in.
- **Migration decision (Wave 2):** transcendentals are NOT transcribed from the
  legacy ad-hoc polynomial. Per the migration plan's spec-mining rule, their
  coefficients come from an **authoritative source (Cephes)** and are verified
  against an f64 oracle at a documented ULP bound (see
  `docs/kernel_specs/*_spec.md`). Legacy is the *structural* reference only.
  Consequence: these kernels are gated by **parity-vs-oracle**, not
  parity-vs-legacy, and their bits differ from legacy by design.
- **Severity:** low-medium (accuracy improves; no silent masking).
- **Post-Wave-6 note:** the oracle-based bound is the intended permanent gate.

---

*Add an entry here the moment a new divergence is found in Waves 2–5. Do not
"fix" any of them during migration — record, defer, review once after teardown.*
