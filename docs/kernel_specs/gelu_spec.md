# gelu — kernel spec (f32, f16/bf16 via f32, i8 LUT)

Follows `docs/kernel_specs/README.md`. **tanh approximation** (decided in
README §4; legacy is tanh-approx). Builds on the validated tanh core.

## Definition
```
gelu(x) = 0.5·x·(1 + tanh( K·(x + C·x³) ))
K = √(2/π) = 0.7978845608,  C = 0.044715
```

## Implementation
The SIMD tiers reuse the arch's validated `tanh8`/`tanh4`/`tanh16`:
`inner = K·(x + C·x³); gelu = 0.5·x·(1 + tanh_core(inner))`. Scalar uses
`f32::tanh`. i8 = 256-entry LUT transcribed VERBATIM (note: legacy i8 LUT uses
K=0.79788456 and clamp(-128,127); kept as-is, Rule 1).

## Edge cases (match torch `approximate='tanh'`, verified)
| x     | inner | tanh | gelu | torch |
|-------|-------|------|------|-------|
| 0     | 0     | 0    | 0    | 0     |
| +inf  | +inf  | 1    | 0.5·inf·2 = inf | inf |
| -inf  | -inf  | -1   | 0.5·(-inf)·0 = NaN | NaN |
| -100  | huge- | -1   | 0.5·(-100)·0 = -0.0 | -0.0 |
| NaN   | NaN   | NaN  | NaN  | NaN   |

torch flushes the deep negative tail to -0 (tanh saturates to exactly -1, so
1+(-1)=0); our tanh core does the same → bit-for-bit torch.

## Tolerance & oracle
- Primary reference: `torch.nn.functional.gelu(x, approximate='tanh')` — MUST use
  approximate='tanh' (the default erf form would fail on a definition mismatch,
  not a bug). scipy cross-check: `0.5·x·(1+scipy.special.erf(...))` is the erf
  gelu — NOT applicable to the tanh variant; skip for gelu (use torch).
- Rust oracle: the f64 tanh-approx formula, rounded to f32. Bound: **≤ 4 ULP**
  (composes tanh + 3 mults + the cube; error accumulates). Edges match torch.

## Surfaces & tiers
- In-place (legacy form; msts uses it) + out-of-place.
- f32: scalar / sse2 / avx1 / avx2 / avx512 / neon (each reuses its tanh core).
- f16/bf16: convert-through-f32. New thresholds GeluF16/Bf16 (GeluF32 exists,
  legacy 64_000).
