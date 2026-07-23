# sigmoid — kernel spec (f32, f16/bf16 via f32, i8 LUT)

Follows `docs/kernel_specs/README.md`. sigmoid builds directly on the validated
exp kernel: **it reuses the exp vector core, no new polynomial**.

## Definition
`sigmoid(x) = 1 / (1 + e^(-x))` (the logistic function).

## Implementation
- SIMD tiers compute `e = exp_core(-x)` (reusing the arch's validated `exp8`/
  `exp4`/`exp16` from `unary/exp/fp32/`) then `1 / (1 + e)` via an accurate
  divide (not `rcp`). Because the exp core already applies its edge masks, all
  sigmoid edge cases fall out for free (no extra masks).
- Scalar tier: `1.0 / (1.0 + (-x).exp())` (legacy's scalar form; std exp).
- f16/bf16: convert-through-f32 (whole buffer, as legacy), then f32 Tier II.
- i8: 256-entry lookup table, transcribed VERBATIM from
  `cpu_old/ops/unary/sigmoid/mod.rs` (`127/(1+exp(-x/16))`, rounded). Rule 1.

## Edge cases (EXACT, inherited from exp_core)
| input       | e = exp(-x) | sigmoid | note |
|-------------|-------------|---------|------|
| `x = +inf`  | exp(-inf)=0 | 1/(1+0)=1 | |
| `x = -inf`  | exp(+inf)=inf | 1/(1+inf)=0 | |
| `x = NaN`   | NaN         | 1/(1+NaN)=NaN | |
| `x = 0`     | 1           | 1/2 = 0.5 | exact |

## Tolerance & oracle
- Oracle: `1/(1+exp(-x))` computed in f64, rounded to f32 (README §1).
- Cross-check: `scipy.special.expit` (this IS a scipy special function) and
  `torch.sigmoid`.
- Bound: **≤ 2 ULP** (one add + one divide over the ≤1-ULP exp core).

## Surfaces & tiers
- Surfaces: in-place (legacy's form; msts uses it) + out-of-place.
- f32 tiers: scalar / sse2 / avx1 / avx2 / avx512 / neon (each reuses its exp core).
- New thresholds SigmoidF32/F16/Bf16 (i8 LUT is cheap, no rayon needed but gated
  uniformly for consistency).

## Verification
- Rust: `#[test]` vs the f64 oracle at ≤2 ULP over a dense sweep + edge table,
  for scalar and every SIMD tier the host supports.
- Python: `tests/parity/test_sigmoid.py` vs `scipy.special.expit` AND `torch.sigmoid`.
