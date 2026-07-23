# tanh — kernel spec (f32, f16/bf16 via f32, i8 LUT)

Follows `docs/kernel_specs/README.md`. Cephes tanhf — NOT `2·sigmoid(2x)-1`,
which suffers catastrophic cancellation near 0 (tanh(x)≈x but 2·0.5-1 loses all
precision). Two branches.

## Source (Cephes single/tanhf.c, fetched 2026-07)
```
threshold = 0.625
small-x poly (Horner on z=x*x):
  P0 = -5.70498872745E-3
  P1 =  2.06390887954E-2
  P2 = -5.37397155531E-2
  P3 =  1.33314422036E-1
  P4 = -3.33332819422E-1
```

## Algorithm
```
ax = |x|
if ax < 0.625:                       # small-x, no cancellation
    z = x*x
    tanh = ((((P0*z+P1)*z+P2)*z+P3)*z+P4)*z*x + x
else:                                # large-x
    s = exp(2*ax)                    # reuses the validated exp core
    tanh = copysign(1 - 2/(s+1), x)
```
Saturation is automatic: for `ax > 0.5*MAXLOGF`, `exp(2*ax)` overflows to +inf in
the exp core, so `1 - 2/(inf+1) = 1`, and copysign gives ±1 — no explicit branch.

## Edge cases (EXACT)
| x     | result | mechanism |
|-------|--------|-----------|
| +inf  | 1      | exp(inf)=inf → 1 → copysign(+) |
| -inf  | -1     | copysign(-) |
| 0     | 0      | small branch: poly·0·0 + 0 = 0 |
| NaN   | NaN    | propagates through both branches; mask picks NaN |

## Tolerance & oracle
- Oracle: `tanh` in f64, rounded to f32 (README §1). Cross-check `np.tanh` and
  `torch.tanh`. Bound: **≤ 2 ULP**.
- Scalar tier uses `f32::tanh()` (std, ~1 ULP), the semantic reference; SIMD
  tiers use the Cephes two-branch above.

## Surfaces & tiers
- In-place (legacy form; msts uses it) + out-of-place.
- f32: scalar / sse2 / avx1 / avx2 / avx512 / neon (each reuses its exp core for
  the large branch; sse2 uses and/andnot/or selects, avx+ use blendv/k-mask).
- f16/bf16: convert-through-f32; i8 = 256-entry LUT (`tanh(x/32)*127`) verbatim.
- New thresholds TanhF32/F16/Bf16.
