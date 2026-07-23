# VERIFIED SWAR Technique Catalog for OxTorch (No-SIMD CPUs & Sub-Byte Fast Paths)

> **Provenance header (added on ingest into repo):**
> - **Source:** deep research report ("VERIFIED SWAR Technique Catalog"),
>   produced by Sonnet-class deep research, supplied by the user.
> - **Ingested:** 2026-07-23, at migration HEAD `9689fe0`.
> - **Status:** SPOT-CHECK REQUIRED BEFORE IMPLEMENTATION. Per project protocol
>   (migration addendum §2), three sources must be independently re-verified
>   before any card here is implemented (Stage 1 / Krok B): musl `HASZERO` (E1),
>   A2 wrapping-sub formula (Chessprogramming wiki), RTN Eq. 8 (C4a,
>   arXiv:1912.02057 §3.4). This status line is updated to "spot-check done:
>   <results>" once that runs. Until then, do NOT implement from this catalog.
> - **Cards flagged partially-unconfirmed by the catalog itself** (treat per the
>   catalog's own guidance): arXiv:2601.11660 subtractive notation; TAB
>   (DOI 10.1145/3508390) reconstruction eq (paywalled). For C4b rely on
>   US Patent 12,033,070 Eq. 7, not the paper notation.

---

## TL;DR
- Partitioned 8×i8/u8 arithmetic (wrapping add/sub, average, abs, min/max, compare→mask) and the ternary popcount dot-product all have **primary, checkable sources** — Warren, *Hacker's Delight* 2nd ed. §2-18/§2-19; Sean Eron Anderson's Stanford "Bit Twiddling Hacks"; the Chessprogramming wiki SWAR pages; musl `src/string/strlen.c`; and RTN (AAAI 2020, DOI 10.1609/aaai.v34i04.5912) / BinaryNet / XNOR-Net — and are safe to implement with the correctness arguments below.
- Two operations the brief asks about have **no clean published word-SWAR form and are flagged as gaps**: general signed *saturating* add/sub across 8 packed bytes with a single simple formula (clean saturating primitives in the literature are hardware MMX/SSE2/VNNI, not word-SWAR), and true i8×i8→i8 lane multiplication. Do NOT synthesize these — widen to 16-bit or fall back to scalar.
- On x86_64 essentially none of these beat SSE2 (`paddb`/`pcmpeqb`/`pavgb`/`pabsb`); this SWAR tier exists for **RISC-V without the V extension, ARMv7 without NEON, and older MIPS**, and as a portable **sub-byte (2-bit/ternary) fallback** where even SSE2 has no direct instruction.

## Key Findings
1. **The carry-blocking add/sub family is fully specified and production-proven.** The canonical `H = 0x80…80`, `L = 0x01…01` masking add/sub appears verbatim on the Chessprogramming wiki "SIMD and SWAR Techniques" page and derives directly from Warren, *Hacker's Delight* §2-18 "Multibyte Add, Subtract, Absolute Value." Both are checkable primary sources and agree.
2. **The signed/unsigned confusion that caused the prior bug is avoidable because the wrapping-add formula is sign-agnostic.** Two's-complement wrapping add/sub produce identical bit patterns for signed and unsigned lanes; the interpretation only matters for saturation, comparison, min/max, and average — which is exactly where this catalog splits signed/unsigned explicitly.
3. **The ternary {-1,0,+1} dot product has two distinct, separately-published encodings** (sign-magnitude two-plane, RTN; and positive/negative subtractive two-plane). Their reconstruction formulas differ, so the audit of `swar.rs` must first determine *which* the file implements.
4. **Sign-bit domain tricks on f16/bf16 bit patterns are exactly correct for neg/abs/copysign/relu-select but NOT for value comparison**, because IEEE sign-magnitude ordering is not monotonic in the raw bit pattern (−0 and NaN break naive integer compare).
5. **Parallel popcount and horizontal byte-sum-by-multiply are textbook and exact**; they underpin the ternary dot product on CPUs lacking hardware POPCNT.

## Details — Spec Cards

Constants used throughout Group A: `H = 0x8080808080808080`, `L = 0x0101010101010101`.

### GROUP A — Partitioned Arithmetic (8×i8/u8 in u64)

---
**A1. Wrapping add (8 lanes), carry-blocked**
- Formula (u64): `z = ((x & ~H) + (y & ~H)) ^ ((x ^ y) & H)`
- Source: Chessprogramming wiki, "SIMD and SWAR Techniques" (SWAR add), https://www.chessprogramming.org/SIMD_and_SWAR_Techniques ; derived from Warren, *Hacker's Delight*, 2nd ed., §2-18, pp. 40–41.
- Correctness: The low 7 bits of each lane are summed with the high bit masked out (`& ~H`), so no carry can cross a lane boundary — the boundary bit is a deliberate "hole." The high bit of each lane is then reconstructed independently as the add-mod-2 (XOR) of the two operand high bits with the carry that propagated *into* that bit from the low-7-bit add: `(x ^ y) & H`. Because the boundary bit is never allowed to carry outward, lane N cannot corrupt lane N+1. This is precisely the invariant the prior hand-rolled adder violated.
- Domain: Full range, all 256×256 lane input pairs. Identical bit result for signed i8 and unsigned u8 (two's-complement wrapping). Wrapping only.
- Ops: ~6 (2 AND, 1 ADD, 1 XOR, 1 AND, 1 XOR) plus mask loads.
- x86 honesty: SSE2 `paddb` does 16 lanes in one instruction; SWAR loses badly. Use only on no-SIMD targets.
- Test: Exhaustive per lane — all 2^16 pairs against `(a as i8).wrapping_add(b as i8)` for each lane position; verify lane independence by fuzzing full u64 pairs.

---
**A2. Wrapping sub (8 lanes), borrow-blocked**
- Formula (u64): `z = ((x | H) - (y & ~H)) ^ ((x ^ ~y) & H)`
- Source: Chessprogramming wiki, "SIMD and SWAR Techniques" (SWAR sub), same URL as A1; Warren §2-18.
- Correctness: Setting the high bit of each minuend lane (`x | H`) guarantees each lane's subtraction cannot borrow across the boundary — the pre-set boundary bit absorbs any borrow. The high bit is then fixed with `(x ^ ~y) & H`, which reconstructs the correct high bit (add-mod-2 of the operand high bits and the borrow-in). No inter-lane leakage.
- Domain: Full range; identical for signed/unsigned wrapping.
- Ops: ~6 plus mask loads.
- x86 honesty: Beaten by SSE2 `psubb`.
- Test: Exhaustive per lane vs `wrapping_sub`.

---
**A3. Unsigned average (round-down), no overflow**
- Formula (u64): `z = (x & y) + (((x ^ y) & ~L) >> 1)`
- Source: Chessprogramming wiki "SIMD and SWAR Techniques" (SWAR average, based on `x+y = (x^y)+2(x&y)`); scalar identity is Warren §2-5 "Average of Two Integers." Raymond Chen, "On finding the average of two unsigned integers without overflow," *The Old New Thing*, Feb 7 2022 (devblogs.microsoft.com/oldnewthing/20220207-00), gives the scalar SWAR form verbatim: `return (a & b) + (a ^ b) / 2;` — the per-lane version masks `~L` before the shift to keep it lane-local.
- Correctness: `x+y = (x^y) + 2·(x&y)`, so `(x+y)>>1 = (x&y) + ((x^y)>>1)`. Masking `~L` before the shift clears each lane's LSB so the right shift cannot pull bit 0 of lane N+1 into bit 7 of lane N. This is the per-lane analog of the classic no-overflow average and cannot leak.
- Domain: Unsigned bytes, full range, exact floor average. For signed lanes this computes the unsigned average of the bit patterns, which is NOT the signed average — do not use for i8.
- Ops: ~4.
- x86 honesty: SSE2 `pavgb` averages 16 unsigned lanes in one instruction but **rounds half-up**, whereas this SWAR form rounds down — the results differ by 1 on odd sums; pick deliberately.
- Test: Exhaustive per lane vs `((a as u16 + b as u16) >> 1) as u8`.

---
**A4. Multibyte absolute value (signed lanes)**
- Formula: build a per-lane all-ones mask `m` where the lane is negative (from the sign bit), then conditionally two's-complement negate within lanes; Warren's word sequence (adapt byte masks to `H`): mask the sign bits, spread them to full-lane masks, XOR, and add the mask back — see §2-18 for the exact 8-instruction sequence.
- Source: Warren, *Hacker's Delight*, 2nd ed., §2-18, p. 41 ("complement and add 1 to each byte whose high bit is on"); Warren §2-4 for the scalar branchless `abs`.
- Correctness: Warren's construction builds a per-lane mask from the sign bit and adds 1 only within lanes; the final add "cannot carry across byte boundaries, because the quantity x ⊕ m has a high-order 0 in each byte" (verbatim, §2-18). That high-order-0 invariant is the leak-prevention guarantee.
- Domain: Signed i8. `abs(-128)` overflows to `-128` (0x80), identical to `i8::wrapping_abs` — document this.
- Ops: ~8.
- x86 honesty: SSSE3 `pabsb` does 16 lanes; SWAR only for no-SIMD.
- Test: Exhaustive per lane vs `(a as i8).wrapping_abs()`.

---
**A5. Negation (signed lanes)**
- Formula (u64): `neg(x) = wrapping_sub(0, x)` = A2 with minuend 0: `z = ((0 | H) - (x & ~H)) ^ ((0 ^ ~x) & H)` = `(H - (x & ~H)) ^ (~x & H)`.
- Source: Composition of A2 (Chessprogramming/Warren §2-18); two's-complement negation is `0 − x`.
- Correctness: Follows directly from A2's borrow-blocking argument with `x_min = 0`.
- Domain: Signed i8; `neg(-128) = -128` (wraps), matches `wrapping_neg`.
- Ops: ~5.
- Test: Exhaustive per lane vs `(a as i8).wrapping_neg()`.

---
**A6. Min / max per byte**
- Formula: compute a per-lane "less-than" mask via A8, then branchless merge `min = a ^ ((a ^ b) & lt_mask)`, `max = b ^ ((a ^ b) & lt_mask)`.
- Source: Anderson, "Bit Twiddling Hacks" (Stanford), branchless min/max `r = y ^ ((x^y) & -(x<y))`, https://graphics.stanford.edu/~seander/bithacks.html ; per-byte packaging exists in the `github.com/dans-stuff/swar` Go package (`SelectSmallerBytes`/`SelectLargerBytes`, math.go).
- Correctness: The scalar branchless min/max relies on `-(x<y)` being all-ones or all-zeros; in SWAR the per-lane compare (A8) produces a `0x00`/`0xFF` lane mask, and the merge `a ^ ((a^b) & mask)` selects per lane with no cross-lane interaction because the mask is lane-local. Use the unsigned compare for unsigned min/max, the biased compare for signed.
- Domain: Full range for the chosen signedness of A8.
- Ops: ~ compare cost + 3.
- x86 honesty: SSE2 `pminub`/`pmaxub` (unsigned) and SSE4.1 `pminsb`/`pmaxsb` (signed) each one instruction.
- Test: Exhaustive per lane vs `a.min(b)`/`a.max(b)`.

---
**A7. Compare → byte mask: EQUALITY (0x00/0xFF lanes)**
- Formula: a lane of `x` equals the corresponding lane of `y` iff that lane of `x^y` is zero. Detect zero lanes with the Mycroft/`HASZERO` marker (E1) applied to `x^y`, giving a high-bit-per-equal-lane word; expand each marker to a full `0xFF` lane by `mask = (marker >> 7) * 0xFF` per lane (equivalently `H_marker` spread).
- Source: Zero-byte test — Anderson "Bit Twiddling Hacks," §"Determine if a word has a zero byte," https://graphics.stanford.edu/~seander/bithacks.html ; and musl `HASZERO` in `src/string/strlen.c`, https://git.musl-libc.org/cgit/musl/tree/src/string/strlen.c . Equality = zero-test of XOR.
- Correctness: `x^y` has a zero byte exactly where lanes are equal. `HASZERO(v) = (v − L) & ~v & H` sets the high bit of each zero lane only (the `& ~v` term suppresses the 0x80 false-positive). Every step is masked with `H`/`~H`, so the borrow used to detect zero is contained inside its lane — no cross-lane leak.
- Domain: Full range (XOR removes any sign concern).
- Ops: ~5–7.
- x86 honesty: SSE2 `pcmpeqb` in one instruction, returns 0x00/0xFF directly.
- Test: Exhaustive per lane vs `if a==b {0xFF} else {0x00}`.

---
**A8. Compare → byte mask: LESS-THAN (unsigned; signed via bias)**
- Unsigned (`a < n`): Anderson's `hasless(x,n) = (x − n·L) & ~x & H` marks lanes below `n`; for lane-vs-lane use the same borrow-hole idea against `y`. The `dans-stuff/swar` package exposes `HighBitWhereLess(v, cm)` (0x80 where `v < cm`).
- Signed: XOR both operands with `H` (bias by +128, mapping i8 [-128,127] monotonically onto u8 [0,255]), then apply the unsigned compare.
- Source: Anderson "Bit Twiddling Hacks," §"Determine if a word has a byte less than n" (`hasless`), Stanford URL above; the bias-by-0x80 signed↔unsigned mapping is standard (Warren §2-12 "Comparison Predicates"). The Oracle x86 Assembly Language Reference Manual documents `pcmpgtb` as *signed* greater-than.
- Correctness: `hasless` uses the same carry-isolation invariant as E1 (borrow into the high bit iff low bits are below the threshold; `& ~x & H` isolates and de-false-positives). The signed bias works because adding 128 preserves order; the XOR is lane-local, so no leak.
- Domain: Unsigned form: full u8. Signed form: full i8 after bias. State explicitly which you built.
- Ops: ~5 (unsigned), ~7 (signed, +2 XOR).
- x86 honesty: `pcmpgtb` (signed) one instruction; unsigned needs a bias just like SWAR.
- Test: Exhaustive per lane vs signed/unsigned `<`.

---
**A9. SATURATING add/sub — DOCUMENTED GAP.** See F2. No clean single-formula word-SWAR *signed* saturating add/sub with full i8 validity is published in the primary references. The `dans-stuff/swar` package provides `AddBytesWithMaximum`/`SubtractBytesWithMinimum` (unsigned clamp to 255/0) but this is an unaudited third-party MIT package, not a primary source, and no signed-saturating word form appears in Hacker's Delight or the Chessprogramming wiki.

### GROUP B — Sign-Bit Domain Tricks for IEEE half formats (4×u16 lanes)

Bit layouts (both 16-bit, sign = bit 15). **bf16** = 1 sign, 8 exponent (bias 127), 7 mantissa (a truncation of the top 16 bits of IEEE-754 float32; layout confirmed by JAX `ml_dtypes` docs and multiple IEEE references). **f16/IEEE half** = 1 sign, 5 exponent (bias 15), 10 mantissa. All B-cards touch only the top bit. Mask `S = 0x8000800080008000`.

---
**B1. Negate (neg)**
- Formula (u64, 4 lanes): `neg = x ^ S`
- Source: First-principles IEEE 754 (sign is bit 15).
- Correctness: IEEE 754 negation flips the sign bit for ALL values incl. ±0, ±inf, and NaN. XOR with the sign mask is therefore exactly IEEE negation, bit-for-bit, lane-independent; NaN payload/quiet bit preserved.
- Domain: All bit patterns, exact, incl. NaN and ±0.
- Ops: 1 XOR.
- Test: Exhaustive per lane — all 2^16 patterns vs a flip-bit-15 reference.

---
**B2. Absolute value (abs)**
- Formula (u64): `abs = x & 0x7FFF7FFF7FFF7FFF`
- Source: First-principles IEEE 754 sign-magnitude.
- Correctness: Clearing the sign bit yields the magnitude for every finite value; ±0→+0, ±inf→+inf, NaN→NaN with sign cleared (IEEE `abs` also just clears sign). Exact, lane-local.
- Domain: All patterns, exact incl. NaN.
- Ops: 1 AND.
- Test: Exhaustive per lane vs clear-bit-15 reference.

---
**B3. Copysign**
- Formula (u64): `copysign(mag, sgn) = (mag & 0x7FFF…) | (sgn & S)`
- Source: First-principles IEEE 754-2008 §5.5.1 `copySign` (sign-bit-only, non-signaling).
- Correctness: Magnitude bits from `mag`, sign bit from `sgn`; matches IEEE `copySign` exactly for all inputs incl. NaN. Lane-local.
- Ops: ~3.
- Test: 2^16 magnitudes × 2 sign choices per lane (full coverage by construction).

---
**B4. ReLU as sign-test + select**
- Formula (u64): spread each lane's sign bit to a full-lane mask `m` (`m = (x & S); m = m - (m >> 15)`), then `relu = x & ~m`.
- Source: Composition of the B-layout sign test with Anderson's branchless conditional select (`w ^= (-f ^ w) & m`), Stanford "Bit Twiddling Hacks."
- Correctness: For any negative lane (sign bit 1, incl. −0.0 and −NaN) the lane is zeroed → +0.0; non-negative lanes pass through. The sign-spread subtraction `m - (m>>15)` operates within the 16-bit field, so no cross-lane leak.
- Domain: **Exactly correct for all finite values and +inf.** TWO documented divergences from IEEE/PyTorch ReLU: (a) `-0.0 → +0.0` (matches `max(x,0)`, generally desirable); (b) **negative NaN is flushed to +0.0**, whereas IEEE `max(NaN,0)`/PyTorch `relu(NaN)` returns NaN. This changes NaN semantics — flag prominently. Positive NaN passes through unchanged.
- Ops: ~4.
- Test: Exhaustive per lane vs a "if sign bit set → +0 else passthrough" reference; add a separate test asserting the intended NaN divergence.

---
**B5. Sign-aware value comparison — DIVERGENT (partial gap, see F5).** There is no exact one-XOR reinterpretation making raw f16/bf16 bit patterns integer-comparable: sign-magnitude is non-monotonic as an unsigned integer (negatives sort in reverse; ±0 tie). The standard fix is a monotonic-key transform, but applied blindly it mis-orders NaN. A clean, primary-sourced *lane-parallel* half-float compare formula was not located, so implement per-lane scalar compare or convert to f32.

### GROUP C — Population Count, Horizontal Sum, and Ternary Dot Product

---
**C1. Parallel population count of a u64 (no hardware POPCNT)**
```
x = x - ((x >> 1) & 0x5555555555555555);
x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;
count = (x * 0x0101010101010101) >> 56;
```
- Source: Anderson, "Bit Twiddling Hacks," §"Counting bits set, in parallel" (Stanford, URL above); Chessprogramming wiki "Population Count." Anderson attributes the multiply-accumulate variant to the AMD Software Optimization Guide; the technique traces to HAKMEM (MIT AI Memo 239, 1972). Wikipedia's "SWAR" article states the final step verbatim: "the last three shift-and-add steps can be combined into `population_count = (x8 * 0x0101010101010101) >> 56`," noting "A multiplication can usually be performed faster" than the shift-add chain.
- Correctness: Counts bits in 2-bit fields, then nibbles, then bytes; the final multiply by `L` sums the eight byte-counts into the top byte (each byte-count ≤ 8, so the accumulation never overflows a byte). Exact for all inputs.
- Ops: ~12.
- x86 honesty: `POPCNT` (SSE4.2) is one instruction — use Rust `count_ones()` (lowers to POPCNT / ARM `cnt`) whenever available; this SWAR form is the fallback when neither POPCNT nor NEON `cnt` exists.
- Test: 2^64 is infeasible; test against the trusted `u64::count_ones()` oracle over millions of random and structured inputs (all-zeros, all-ones, single bits, byte patterns).

---
**C2. Horizontal byte-sum via multiply by 0x0101…01**
- Formula (u64): if the total of all 8 lanes ≤ 255, `sum = (x * 0x0101010101010101) >> 56`.
- Source: Anderson "Bit Twiddling Hacks" (used inside the popcount multiply step); Lemire, "Bit Hacking (with Go code)," 2023-02-07, https://lemire.me/blog/2023/02/07/bit-hacking-with-go-code/ .
- Correctness: Multiplying by `L` accumulates all eight byte-lanes into the top byte (each partial product shifts a lane into the MSB position). Valid **only if the total ≤ 255**, else the top byte overflows and silently wraps — a hard validity bound.
- Domain: Lane sum ≤ 255. For larger sums, split into two halves or widen.
- Ops: 2 (multiply + shift).
- Test: Exhaustive over lane configurations whose sum ≤ 255; assert wrap behavior above the bound.

---
**C3. Binary {-1,+1} dot product via XOR/XNOR + popcount**
- Formula: encode ±1 as bits (1→+1, 0→−1). Then `dot = N − 2·popcount(a XOR b) = 2·popcount(a XNOR b) − N`, N = number of lanes/bits.
- Source (PRIMARY): Courbariaux, Hubara, Soudry, El-Yaniv & Bengio, "Binarized Neural Networks," arXiv:1602.02830 (2016) — origin of the XNOR+bitcount MAC replacement; Rastegari, Ordonez, Redmon & Farhadi, "XNOR-Net," ECCV 2016, arXiv:1603.05279 — binary convolution with scaling factors. The explicit closed form is stated verbatim in "Zeros can be Informative" (arXiv:2601.11660, Eq. 1): `a·b = Σ(2aᵢ′−1)(2bᵢ′−1) = 2·popc(a′ XNOR b′) − n`.
- Correctness: Each bit agreement contributes +1, each disagreement −1; popcount of XNOR counts agreements `p`, so the signed sum is `p − (N−p) = 2p − N`. Exact.
- Ops: 1 XOR + popcount(C1) + 2.
- x86 honesty: With `POPCNT` this is genuinely fast and used in production BNN kernels; without it, uses C1.
- Test: Exhaustive for small N against an integer reference (feasible for N ≤ 8 per byte-lane).

---
**C4. Ternary {-1,0,+1} dot product — TWO published two-plane encodings (critical for the `swar.rs` audit)**

**C4a. Sign-magnitude two-plane (RTN scheme).**
- Encoding: each ternary value is two bits — bit A = "is-nonzero" (magnitude/mask), bit B = sign. Weight planes `(W1 = nonzero mask, W2 = sign)`, activation planes `(A1, A2)`.
- Reconstruction (verbatim RTN Eq. 8): let `c = W1 AND A1` (positions where both operands nonzero). Then
```
dot = popcount(c) - 2 * popcount( (W2 XOR A2) AND c )
```
- Source (PRIMARY, confirmed verbatim): Li, Dong, Zhang, Bai, Chen & Wang, "RTN: Reparameterized Ternary Network," AAAI 2020, Vol. 34(04), pp. 4780–4787, DOI 10.1609/aaai.v34i04.5912 (arXiv:1912.02057), §3.4 "Efficient Computation Pattern," Table 1 and Eq. 8. Encoding text verbatim: "the first bit indicates whether this number is zero or not, and the second bit indicates the sign of this number"; "zero can be represented by either 00 or 01." The paper reports its FPGA computation pattern "brings 46.46x and 89.17x savings on power and area respectively compared with the full precision convolution."
- Correctness: `popcount(c)` counts positions where both operands are nonzero (each product is ±1). Within those, `W2 XOR A2 = 1` marks sign disagreement (product −1); each such position flips a +1 to −1, i.e. subtracts 2. So `dot = (#nonzero-overlap) − 2·(#sign-mismatch)`. Exact. The `×2` is a left shift.
- Sign convention: RTN Table 1 uses sign bit 1→+1, 0→−1. **Verify `swar.rs` matches this convention** — an inverted sign bit silently negates every product.

**C4b. Positive/negative subtractive two-plane.**
- Encoding: `w = w_pos − w_neg`, with `w_pos, w_neg ∈ {0,1}` bit-planes (w_pos set where w=+1, w_neg set where w=−1, both 0 where w=0). Binary activation as bits `a`.
- Reconstruction: `dot = popcount(w_pos AND a) − popcount(w_neg AND a)`. Confirmed as a primary datapoint in US Patent 12,033,070, Eq. 7: `Z = POPCOUNT(AND(X, W⁺)) − POPCOUNT(AND(X, W⁻))`. The "subtractive bit-encoding" using two bit-planes combined via XOR+popcount is also the method of "Zeros can be Informative" (arXiv:2601.11660, Algorithm 1).
- Correctness: `w_pos AND a` counts +1 contributions, `w_neg AND a` counts −1 contributions; their difference is the signed dot. Exact for binary activations; for ternary activations use the RTN form (C4a) or extend with an activation nonzero-mask.
- Audit guidance: **The two encodings are NOT interchangeable.** If `swar.rs` stores (mask, sign) planes it must use C4a's `popcount(c) − 2·popcount((W2⊕A2)&c)`. If it stores (pos, neg) planes it must use C4b's difference of two ANDed popcounts. Applying C4a's formula to pos/neg planes (or vice versa) yields silently wrong sums with no crash — this is the highest-risk audit item.

**Production honesty note.** State-of-the-art ternary LLM CPU kernels do NOT use popcount. Microsoft's bitnet.cpp uses Ternary Lookup Table (TL1/TL2) and "Int2 with a Scale" (I2_S) kernels; per the paper's Tables 5–6 verbatim, "TL1 Kernel transforms every two full-precision weights into 4-bit index" and "TL2 Kernel compresses every three full-precision weights into a 1-bit sign (0 or 1) and a 4-bit index." Source: Wang et al., "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs," ACL 2025, pp. 9305–9322, DOI 10.18653/v1/2025.acl-long.457 (arXiv:2502.11880); the paper reports the TL/I2_S kernels achieve "up to a 6.25x increase in speed over full-precision baselines and up to 2.32x over low-bit baselines," with TL2_0 outperforming T-MAC "by up to 2.32x on Intel i7-13700H and 1.19x on Apple M2 Ultra." TQ2_0 in llama.cpp uses 2-bit packing (2.0625 bpw) with multiply-add. The Litespark paper (arXiv:2605.06485, May 2026) stores weights as int8 and feeds NEON `SDOT` / AVX-512 VNNI (`VPDPBUSD`-family) int8 dot-product instructions directly, explicitly avoiding 2-bit packing + popcount. **The popcount two-plane path (C4) is therefore appropriate for OxTorch's no-SIMD / no-VNNI targets, not as an x86 fast path.**

### GROUP D — Sub-Byte Unpacking

---
**D1. Byte broadcast (build a splat constant)**
- Formula: `splat = (byte as u64) * 0x0101010101010101`
- Source: Lemire, "Bit Hacking (with Go code)," 2023-02-07 (URL above): "0x12 * uint64(0x0101010101010101) == 0x1212121212121212."
- Correctness: Multiply replicates the low byte into all 8 lanes; exact for any byte.
- Ops: 1 multiply.
- Test: Exhaustive over all 256 bytes.

---
**D2. 4-bit nibble spread → 8-bit lanes**
- Formula: low nibbles `x & 0x0F0F0F0F0F0F0F0F`; high nibbles `(x >> 4) & 0x0F0F0F0F0F0F0F0F`; then place into separate byte lanes.
- Source: The nibble-mask idiom is standard; Lemire's "SWAR explained: parsing eight digits," https://lemire.me/blog/2022/01/21/swar-explained-parsing-eight-digits/ , demonstrates the mask/shift/multiply decomposition pattern, and Anderson "Bit Twiddling Hacks" covers deposit/interleave.
- Correctness: Nibble masks `0x0F0F…` are lane-local; shifting by 4 within a byte cannot cross byte boundaries after masking. Exact.
- Ops: ~3–4.
- Test: Exhaustive over all 2^16 for a 2-nibble field.

---
**D3. 2-bit field → 8-bit lanes**
- Formula: for 2-bit fields packed 4-per-byte, extract with `(x >> shift) & 0x03` per field (or spread with mask `0x0303…` plus shifts).
- Source: Lemire, "SWAR explained: parsing eight digits" (mask/multiply/shift decomposition, URL above); Anderson "Bit Twiddling Hacks" for deposit/gather. **NOTE:** a single-multiply 2-bit→8-bit expander with a fully verified constant was NOT located in a primary source; the robust published approach is mask+shift per field. If OxTorch wants a one-multiply expander, the constant must be derived and exhaustively tested in-house (do not assume a constant from memory).
- Correctness: `& 0x03` masks are lane-local; exact.
- Ops: ~2 per field (mask+shift), or fewer with a verified multiply constant.
- Test: Exhaustive over all 256 byte values (all 2-bit field configurations).

### GROUP E — Scanning Primitives

---
**E1. haszero / hasvalue byte detection (Mycroft's trick, as used in musl)**
- Formula: `#define HASZERO(x) (((x) - ONES) & ~(x) & HIGHS)` with `ONES = 0x0101010101010101`, `HIGHS = 0x8080808080808080`; `hasvalue(x, n) = HASZERO(x ^ (ONES * n))`.
- Source (PRIMARY, production): musl libc, `src/string/strlen.c`, `HASZERO` macro, https://git.musl-libc.org/cgit/musl/tree/src/string/strlen.c (mirror: github.com/bminor/musl). Attributed to Alan Mycroft (1987). Also Anderson "Bit Twiddling Hacks," §"Determine if a word has a zero byte."
- Correctness: Subtracting `ONES` borrows into the high bit of a lane iff that lane's low 7 bits are zero; `& ~x` requires the lane's high bit was also 0; `& HIGHS` isolates the marker. A lane's high bit in the result is set iff the byte was zero. **Caveat (from Anderson):** the *pretest* fast form has occasional false positives when a lane is 0x80; musl's `HASZERO` as written is exact because the `& ~x` term suppresses that case. Lane-local: the borrow is contained by the `& ~x & HIGHS` masking.
- Domain: Full range for zero detection; `hasvalue` exact for a specific byte `n`.
- Ops: ~4 (haszero), ~6 (hasvalue with splat).
- x86 honesty: SSE2 `pcmpeqb` + `pmovmskb` is faster where available; this is the portable scalar path musl uses on all ISAs (its correctness is battle-tested).
- Test: Exhaustive per lane (2^8 for zero; 2^16 for hasvalue pairs).

---
**E2. Byte broadcast** — see D1 (`× 0x0101…01`).

---
**E3. First / last set byte extraction**
- Formula: given a per-lane marker word (high bit set in matching lanes, e.g. from E1 or A7), first matching byte index = `trailing_zeros(marker) >> 3`; last = `(63 − leading_zeros(marker)) >> 3` (little-endian: first byte = lowest).
- Source: Richard Startin, "Finding Bytes in Arrays," https://richardstartin.github.io/posts/finding-bytes (documents the `numberOfLeadingZeros(tmp) >>> 3` byte-index idiom derived from *Hacker's Delight* ch. 6); Warren §5-4 "Counting Trailing 0's" / ch. 6 zero-byte search. Rust `u64::trailing_zeros`/`leading_zeros` lower to `TZCNT`/`LZCNT`/ARM `clz`.
- Correctness: Each lane's marker sits at a known bit offset (bit 7 of each byte after E1); `tzcnt >> 3` converts bit index to byte index. Exact given the marker invariant.
- Ops: 1 count + 1 shift (plus marker construction).
- Test: Exhaustive over single-lane-set markers (8 positions) and multi-set combinations.

### GROUP F — Negative Results (operations NOT to attempt in SWAR)

**F1. General i8×i8 → i8 lane multiplication.** No published word-SWAR method produces correct per-byte products in-place. First-principles argument: an 8×8 product needs up to 16 bits, so adjacent lanes' products overlap and there is no masking that separates them within one machine multiply. Published consensus (general SIMD practice; Grokipedia SWAR entry) is to **widen to 16-bit lanes or use hardware `pmullw`/table lookup**. Do not synthesize — widen to 4×i16 or fall back to scalar.

**F2. Signed saturating add/sub across 8 packed bytes.** No single clean word-SWAR formula with full i8 validity is in the primary references (Hacker's Delight, Chessprogramming wiki). *Unsigned* saturating add/sub can be built (wrapping sum via A1, detect overflow from the blocked carry high-bit, then OR-in 0xFF / mask to 0), and `dans-stuff/swar` implements the unsigned clamp — but that is not a primary-sourced construction, and the *signed* case (exactly where the prior bug lived) is materially harder. **Recommendation: implement unsigned-saturating with in-house exhaustive tests; treat signed-saturating as a gap — widen to i16, saturate, narrow.** All clean saturating primitives that exist (`_mm_adds_epi8`, MMX `padd*.uus`, WASM `*_add_sat`, VNNI) are hardware, not word-SWAR.

**F3. Any transcendental / division on packed lanes.** No SWAR form; requires per-lane iteration or table lookup. Division by a runtime value has no word-parallel bit trick (Anderson's modulus tricks apply only to divisors of the form `2^s` or `2^s − 1`). Do not attempt.

**F4. f16/bf16 arithmetic beyond sign-domain ops.** Addition/multiplication of half-floats requires exponent alignment, mantissa multiply/add, normalization, and rounding — none expressible as lane-parallel word bit-ops. Only neg/abs/copysign/relu-select (Group B) are exact bit-level operations. For actual f16/bf16 add/mul, convert to f32 (or use hardware f16): bf16→f32 is a zero-extend of the mantissa; f16→f32 needs an exponent rebias.

**F5. Lane-parallel f16/bf16 value comparison.** As in B5, no primary-sourced word-parallel monotonic-key compare for half-floats was located; NaN and −0 break naive integer comparison of sign-magnitude patterns. Implement per-lane scalar compare or convert to f32.

## Recommendations

**Stage 1 — implement the fully-sourced, exhaustively-testable core now (lowest risk):** A1 (wrapping add), A2 (wrapping sub), A5 (neg), A4 (abs), A3 (unsigned average), A7/A8 (compare masks), B1/B2/B3 (bf16/f16 neg/abs/copysign), C1 (popcount), C2 (horizontal sum), D1 (broadcast), E1 (port the exact musl `HASZERO` macro). Each is byte-exhaustively testable (2^16 pairs/lane). Gate merge on a 100% exhaustive pass.

**Stage 2 — implement with extra scrutiny:** A6 (min/max, depends on a correct A8 compare), B4 (relu — ship WITH the documented NaN-flush divergence in a doc comment and a test that asserts it), C3 (binary dot), C4a/C4b (ternary dot). For C4, **before writing any formula, audit `cpu_old/ops/bitnet/swar.rs` to determine which encoding it uses** (sign-magnitude vs pos/neg planes) and confirm the sign-bit convention against RTN Table 1; then apply exactly the matching reconstruction (C4a or C4b). Add a golden test with a hand-computed ternary vector whose result is nonzero and sign-sensitive (e.g. mostly-negative products) — the class of input that exposed the prior carry bug.

**Stage 3 — gaps, do NOT ship SWAR versions:** i8×i8 (F1), signed saturating (F2), transcendentals/division (F3), f16 arithmetic (F4), f16 compare (F5). Route these to widen-compute-narrow or scalar fallbacks.

**Benchmark thresholds that change the plan:** On any target where `count_ones()` compiles to a hardware POPCNT/`cnt`, prefer it over C1. On any target with SSE2/NEON, compile the entire Group A SWAR tier out in favor of intrinsics — keep SWAR strictly behind a `#[cfg(not(target_feature = ...))]` gate. If a target gains VNNI/`SDOT`, switch ternary to int8-dot (the Litespark approach) and retire the C4 popcount path.

## Caveats
- **Third-party vs primary:** the `dans-stuff/swar` Go package is a convenient index of operation *names* and confirms which ops are considered feasible, but it is an unaudited MIT package, not a primary source; every formula cited here ultimately traces to Hacker's Delight, the Stanford "Bit Twiddling Hacks" page, the Chessprogramming wiki, musl, Lemire's blog, or the named papers.
- **Hacker's Delight page/section numbers** are from the 2nd edition (2013, ISBN 978-0321842688): §2-4 abs, §2-5 average, §2-12 comparison predicates, §2-18 multibyte add/sub/abs (pp. 40–41), §2-19 doz/max/min, §5-1 counting 1-bits, §5-4 counting trailing 0s. Verify against your copy before quoting figure numbers in code comments.
- **Ternary encoding is the top audit risk:** the two published schemes (C4a RTN sign-magnitude vs C4b pos/neg subtractive) use different formulas and sign conventions; applying the wrong one produces silently wrong (not crashing) dot products. The `swar.rs` audit must nail down the encoding and sign convention first.
- **Operation counts are approximate** (Anderson's methodology: one C operator = one op; mask loads not always counted). Real instruction counts and whether SWAR ever wins depend on register pressure and the target ISA — benchmark on the actual target; the primary sources give no CPU-time measurements for these no-SIMD paths.
- **NaN semantics divergence in ReLU (B4)** is a deliberate, documented behavior change, not a bug: negative NaN is flushed to +0.0. If bit-exact IEEE `max(x,0)` NaN propagation is required, do not use the SWAR relu.
- **The XNOR/popcount binary-dot origin** is split between BinaryNet (arXiv:1602.02830, the bitcount MAC) and XNOR-Net (arXiv:1603.05279, binary conv + scaling); the clean `2·popcount(XNOR) − N` closed form is confirmed verbatim in arXiv:2601.11660 Eq. 1. Cite the primary you actually rely on in code.
- **Two subagent-reported items remain partially unconfirmed** and should be treated cautiously: the exact `bᵢ = bᵢ^pos − bᵢ^neg` notation attributed to arXiv:2601.11660 was not verified verbatim (the *concept* of subtractive two-plane encoding is confirmed there and in US Patent 12,033,070 Eq. 7), and TAB's (ACM TECS, DOI 10.1145/3508390) exact reconstruction equation could not be extracted (ACM page blocked). For the C4b formula, rely on the patent's verbatim `POPCOUNT(AND(X,W⁺)) − POPCOUNT(AND(X,W⁻))` rather than the paper notation.

---

### Consolidated Bibliography (checkable identifiers)
1. Warren, Henry S. *Hacker's Delight*, 2nd ed. Addison-Wesley, 2013. ISBN 978-0321842688. §2-4, §2-5, §2-12, §2-18 (pp. 40–41), §2-19, §5-1, §5-4. Chapter-2 excerpts online at informit.com/articles/article.aspx?p=1959565 (seqNum=18 multibyte add/sub/abs; seqNum=19 doz/max/min).
2. Anderson, Sean Eron. "Bit Twiddling Hacks." Stanford. https://graphics.stanford.edu/~seander/bithacks.html (parallel popcount; branchless min/max; conditional select; zero-byte / hasless byte tests).
3. Chessprogramming wiki. "SIMD and SWAR Techniques." https://www.chessprogramming.org/SIMD_and_SWAR_Techniques (SWAR add/sub/average with `H`/`L`). Also "Population Count," "Parallel Prefix Algorithms."
4. musl libc. `src/string/strlen.c`, `HASZERO` macro. https://git.musl-libc.org/cgit/musl/tree/src/string/strlen.c (Mycroft 1987 zero-byte detection, production).
5. Lemire, Daniel. "Bit Hacking (with Go code)," 2023-02-07, https://lemire.me/blog/2023/02/07/bit-hacking-with-go-code/ ; "SWAR explained: parsing eight digits," 2022-01-21, https://lemire.me/blog/2022/01/21/swar-explained-parsing-eight-digits/ ; "Detect control characters… using SWAR," 2025-04-13.
6. Chen, Raymond. "On finding the average of two unsigned integers without overflow." The Old New Thing, 2022-02-07. devblogs.microsoft.com/oldnewthing/20220207-00.
7. Startin, Richard. "Finding Bytes in Arrays." https://richardstartin.github.io/posts/finding-bytes (byte-index via `clz>>3`).
8. Li, Dong, Zhang, Bai, Chen, Wang. "RTN: Reparameterized Ternary Network." AAAI 2020, 34(04):4780–4787. DOI 10.1609/aaai.v34i04.5912. arXiv:1912.02057. §3.4, Table 1, Eq. 8.
9. Courbariaux, Hubara, Soudry, El-Yaniv, Bengio. "Binarized Neural Networks." arXiv:1602.02830 (2016).
10. Rastegari, Ordonez, Redmon, Farhadi. "XNOR-Net." ECCV 2016. arXiv:1603.05279.
11. Wu, Song, Kondguli, Geng, Li. "Zeros can be Informative: Masked Binary U-Net…" arXiv:2601.11660 (subtractive bit-encoding; Eq. 1 binary dot; Algorithm 1).
12. Wang et al. "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs." ACL 2025, pp. 9305–9322. DOI 10.18653/v1/2025.acl-long.457. arXiv:2502.11880. (TL1/TL2, I2_S; Tables 5–7.) Related: "1-bit AI Infra," arXiv:2410.16144.
13. Dade, Morri, Rahat, Pal. "Litespark Inference on Consumer CPUs." arXiv:2605.06485 (2026) — int8 SDOT/VNNI ternary dot.
14. US Patent 12,033,070 — subtractive two-plane ternary dot, Eq. 7: `POPCOUNT(AND(X,W⁺)) − POPCOUNT(AND(X,W⁻))`.
15. Oracle. *x86 Assembly Language Reference Manual*, §3.13.4 (`pcmpeqb`/`pcmpgtb` signed). docs.oracle.com/cd/E36784_01/html/E36859/epmnx.html.
16. Wikipedia. "SWAR" (framing; popcount-multiply step). JAX `ml_dtypes` bf16 docs (bf16 layout confirmation).
