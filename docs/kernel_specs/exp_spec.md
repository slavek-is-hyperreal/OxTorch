# exp — kernel spec (f32, and f16/bf16 via f32)

Follows `docs/kernel_specs/README.md` (oracle, ULP policy). exp is the foundation
of the transcendental family (sigmoid = 1/(1+exp(-x)); tanh via exp; gelu via
tanh). **Legacy is NOT the numeric reference** (its ad-hoc split `0.6931457519`
is not Cephes); coefficients below are transcribed verbatim from Cephes.

## Definition
`exp(x) = e^x`, computed by range reduction + a degree-5 minimax polynomial.

## Source (provenance — transcribe verbatim, never invent)
Cephes single-precision `expf`, files fetched 2026-07:
- `cephes/single/expf.c` — algorithm + C1, C2, polynomial coefficients.
- `cephes/single/constf.c` — LOG2EF, MAXLOGF, MINLOGF.

```
LOG2EF = 1.44269504088896341
C1     = 0.693359375
C2     = -2.12194440e-4          (note: ln2 = C1 + C2, a Cody-Waite hi/lo split)
poly (Horner, first→last):
  P0 = 1.9875691500E-4
  P1 = 1.3981999507E-3
  P2 = 8.3334519073E-3
  P3 = 4.1665795894E-2
  P4 = 1.6666665459E-1
  P5 = 5.0000001201E-1
MAXLOGF =  88.72283905206835     (exp overflows to +inf above this)
MINLOGF = -103.278929903431851103 (exp underflows to 0 below this)
```

## Algorithm (per Cephes expf)
```
n   = round_to_nearest(LOG2EF * x)      # Cephes uses floor(LOG2EF*x + 0.5);
                                         # SIMD uses cvtps_epi32 (round-nearest),
                                         # equivalent within tolerance — noted.
g   = x - n*C1 - n*C2                    # reduced arg, g in ~[-ln2/2, ln2/2]
p   = ((((P0*g + P1)*g + P2)*g + P3)*g + P4)*g + P5
e^g = p*(g*g) + g + 1.0
e^x = ldexp(e^g, n) = e^g * 2^n         # 2^n built as ((n+127)<<23) reinterpret
```

## Edge cases (EXACT match, not ULP)
Applied by mask AFTER the polynomial (clamp handles the poly domain; masks fix
the ends). `min/max` return the non-NaN operand, so NaN needs its own mask.

| input            | result | mechanism |
|------------------|--------|-----------|
| `x` is NaN       | NaN    | `x != x` select |
| `x = +inf`       | +inf   | caught by `x > MAXLOGF` |
| `x = -inf`       | 0.0    | caught by `x < MINLOGF` |
| `x > MAXLOGF`    | +inf   | overflow select |
| `x < MINLOGF`    | 0.0    | underflow select |
| `x = 0`          | 1.0    | poly is exact at g=0: 0+0+1 |

NOTE (divergence from Cephes-the-library): Cephes clamps overflow to `MAXNUMF`
(largest finite); we return `+inf` to match PyTorch / IEEE `std::exp`. Recorded
implicitly via the edge-case table; the oracle is `std::exp`/f64, which also
gives +inf.

## Tolerance & oracle
- Oracle: `np.exp(x.astype(f64)).astype(f32)` (README §1). Cross-check vs
  `scipy.special` implicitly (same Cephes lineage) and vs `torch.exp`.
- Bound: **≤ 2 ULP** over `x ∈ [MINLOGF, MAXLOGF]`; edge cases exact.
- Scalar tier uses `f32::exp()` (std, ~0.5 ULP) — trivially within bound and is
  the semantic reference. SIMD tiers use the Cephes poly above.

## Surfaces & tiers
- Surfaces: in-place `exp_inplace` (legacy's only form; msts uses it) AND
  out-of-place `exp` (convenience, uniform with relu/neg).
- f32 tiers: scalar / sse2 / avx1 / avx2 / avx512 / neon. sse2 blends edge masks
  with and/andnot/or (no blendv); avx1+ use blendv/cmp.
- f16/bf16: convert-through-f32 (whole buffer via TensorPool, as legacy), then
  the f32 Tier II serial kernel.

## Verification
- Rust: `#[test]` computes the f64 oracle and asserts ≤2 ULP over a dense sweep
  of [MINLOGF, MAXLOGF] plus the edge-case table (exact). Runs for scalar, and
  for every SIMD tier the host supports (force_arch).
- Python: `tests/parity/test_exp.py` vs `np.exp` f64-oracle AND `torch.exp`.
