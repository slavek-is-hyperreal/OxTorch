# Transcendental kernel specs — tolerance policy & conventions

Precedent for the whole transcendental family (exp, sigmoid, silu, tanh, gelu).
Every `docs/kernel_specs/{op}_spec.md` MUST state its numbers explicitly; a
differential test without a written bound "passes or fails by mood".

## 1. The oracle (what we compare against)

The reference is **the true real-valued function computed in f64, then rounded
once to f32** — NOT PyTorch, NOT scipy-in-f64 directly.

```python
def oracle_exp(x_f32):            # example
    return np.exp(x_f32.astype(np.float64)).astype(np.float32)
```

Rationale: this is the best f32 result achievable, so it measures the kernel's
own error, independent of how torch/scipy round. `scipy.special` is used as a
cross-check that our f64 oracle itself is right (it wraps the same Cephes math),
not as the primary reference.

## 2. ULP bounds (per op, binding gate)

Error is measured in **ULP vs the oracle** (`|kernel - oracle|` in units of the
last place of the oracle value). Bounds:

| op      | bound      | note |
|---------|------------|------|
| exp     | ≤ 2 ULP    | the foundation; everything below inherits its error |
| tanh    | ≤ 2 ULP    | |
| sigmoid | ≤ 2 ULP    | = 1/(1+exp(-x)); one add + one div over exp |
| silu    | ≤ 3 ULP    | = x·sigmoid(x); carries sigmoid's error × |x| |
| gelu    | ≤ 4 ULP (or atol 1e-6) | tanh-approx: composes tanh+mul+poly; error accumulates |

Edge cases are **exact-match**, not ULP: `±0`, `±inf`, `NaN`, and the documented
saturation points (e.g. `sigmoid(+inf)=1`, `exp(-inf)=0`). Denormal handling is
stated per spec.

## 3. Legacy is NOT the numeric truth for this family

Rule 1 ("transcribe legacy") is **overridden for transcendentals** by the
migration plan's spec-mining rule: legacy used ad-hoc low-degree polynomials
(and an inconsistent scalar `std::exp` tail), which are not authoritative.
Coefficients come from **Cephes** (public domain), transcribed verbatim and
diffable against the original — *those* are the Rule-1 material. Legacy provides
loop structure / dispatch shape only. Therefore these kernels are gated by
**parity-vs-oracle at the ULP bound above**, not parity-vs-legacy. This is
recorded in `docs/known_divergences.md` §3.

## 4. gelu definition — TANH-APPROX (decided)

Legacy gelu (`cpu_old/ops/unary/gelu/gelu_f32.rs`) is the **tanh approximation**,
not the erf form:

```
gelu(x) = 0.5·x·(1 + tanh( √(2/π)·(x + 0.044715·x³) ))
          K = √(2/π) = 0.7978845608,  C = 0.044715
```

We keep tanh-approx (Rule 1 for the *definition*, even though the *coefficients*
of its inner tanh/exp come from Cephes). **Parity tests MUST call torch with the
matching variant:** `torch.nn.functional.gelu(x, approximate='tanh')` — NOT the
default (erf). Comparing against erf-gelu would fail on a definition mismatch,
not a bug. This ordering is why exp is implemented first: gelu → tanh → exp.

## 5. Spec file format (`{op}_spec.md`)

Each spec states: the exact formula & definition; the Cephes source (file +
identifier) for every coefficient, transcribed verbatim; the range-reduction
scheme and interval bounds; edge-case table (±0/±inf/NaN/denormal → exact
result); the ULP bound (from §2) and the oracle (§1); and the list of arch tiers
implemented.
