# silu (swish) — kernel spec (f32, f16/bf16 via f32, i8 LUT)

Follows `docs/kernel_specs/README.md`. Builds on the validated exp kernel.

## Definition
`silu(x) = x · sigmoid(x) = x / (1 + e^(-x))` (a.k.a. swish, β=1).

## Implementation — match PyTorch, NOT the stable form
The SIMD tiers compute `e = exp_core(-x)` then `x / (1 + e)`, reusing the arch's
validated `exp8`/`exp4`/`exp16`.

**Deliberate choice (verified against torch):** we use the NAIVE `x/(1+exp(-x))`,
NOT `x·stable_sigmoid(x)`. Measured `torch.nn.functional.silu`:
`silu(-100) = -0.0`, `silu(-inf) = NaN`, `silu(+inf)=+inf`, `silu(0)=0`,
`silu(NaN)=NaN`. i.e. torch lets `exp(-x)` overflow in f32 and flushes the deep
negative tail to ∓0. The exp core already saturates `exp(-x)` to +inf for
`x < -MAXLOGF`, so `x/(1+inf) = ∓0` — matching torch bit-for-bit. Using a stable
sigmoid would instead preserve a tiny denormal (`-100·3.7e-44 ≈ -3.7e-42`),
DIVERGING from torch. For a drop-in replacement, matching torch wins.

- Scalar: `x / (1.0 + (-x).exp())` (legacy's form).
- f16/bf16: convert-through-f32 (whole buffer, as legacy).
- i8: 256-entry LUT, `x/16` domain scaling, transcribed VERBATIM. Rule 1.

## Edge cases (match torch, inherited from exp_core)
| x       | e=exp(-x) | silu = x/(1+e) | torch |
|---------|-----------|----------------|-------|
| +inf    | 0         | inf            | inf   |
| -inf    | inf       | -inf/inf = NaN | NaN   |
| -100    | inf       | -100/inf = -0.0| -0.0  |
| 0       | 1         | 0/2 = 0        | 0     |
| NaN     | NaN       | NaN            | NaN   |

## Tolerance & oracle
- Primary reference: `torch.nn.functional.silu` (this is a drop-in target and it
  defines the tail-flush behaviour we match).
- Rust oracle test: vs the f64 naive form `x/(1+exp(-x))` over the
  WELL-CONDITIONED range `|x| ≤ 40` (where f32 and f64 agree; outside it torch
  itself flushes and the f64 oracle is not the reference) at **≤ 3 ULP** (silu
  carries sigmoid's error × |x|). Tail + edges: match torch exactly.

## Surfaces & tiers
- In-place (legacy form; msts uses it) + out-of-place.
- f32: scalar / sse2 / avx1 / avx2 / avx512 / neon (each reuses its exp core).
- New thresholds SiluF32/F16/Bf16.
