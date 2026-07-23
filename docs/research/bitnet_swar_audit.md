# Krok C — audit of `cpu_old/ops/bitnet/swar.rs` (Wave-5 blocker)

Status: **DONE — two findings, Wave 5 (matmul/bitnet/quant) BLOCKED pending a
user decision on finding #2.** Golden test:
`cpu_old::ops::bitnet::swar::krok_c_audit::swar_row_order_vs_packing`.

## Finding 1 — encoding is a THIRD scheme, not C4a/C4b
The catalog (docs/research/swar_catalog.md) offered two ternary dot encodings:
- **C4a** (RTN, arXiv:1912.02057): sign-magnitude two-plane, `popcount(c) −
  2·popcount((W2⊕A2)&c)`, `c = W1∧A1`.
- **C4b** (US Patent 12,033,070): pos/neg two-plane, `popcount(W⁺&X) −
  popcount(W⁻&X)`.

`swar.rs` (and `avx2.rs`, `sse.rs`, `scalar.rs`) implement **neither**. They use a
**2-bit offset-binary** encoding:
- weight `w ∈ {-1,0,+1}` stored as `q = w + 1 ∈ {0,1,2}` (2 bits, 4 rows/byte).
- dot via a plain integer **MAC**: `dot = Σ q·act`, then a bias correction
  `output = (dot − a_sum)·scale` where `a_sum = Σ act`, i.e.
  `Σ (q−1)·act·scale = Σ w·act·scale`. **No popcount, no bit-planes.**

⇒ The C4a/C4b popcount formulas MUST NOT be "ported" onto this code. When bitnet
moves in Wave 5 it stays MAC+bias (move-not-rewrite), and the catalog C4 cards
are irrelevant to it. (The catalog remains correct for a *future* two-plane
popcount path, e.g. on no-VNNI targets — just not for THIS file.)

## Finding 2 — CONFIRMED row-order bug (pack vs compute disagree)
The packer and the compute kernels disagree on which 2-bit field is which row:

| | row0 | row1 | row2 | row3 |
|---|---|---|---|---|
| **pack** (`tensor/conversion.rs::execute_to_bitnet`, BitNet2) | bits[0:1] | [2:3] | [4:5] | [6:7] |
| **compute** (swar/avx2/sse/scalar, all identical) | bits[6:7] | [4:5] | [2:3] | [0:1] |

Packing is `(q0<<0)|(q1<<2)|(q2<<4)|(q3<<6)` ("LSB-first, matches safetensors");
every compute kernel reads row0 as `(byte>>6)&3`. ⇒ within each group of 4 output
rows the rows come out **REVERSED (0↔3, 1↔2)**. Silent wrong result, no crash —
exactly the class Krok C exists to catch.

**Golden proof** (hand-computed, sign-sensitive): weight rows
`[+1×8], [−1×8], [0×8], [−1,+1,…]`, `act = [2,3,−1,5,−4,6,7,−2]`:
- expected (pack row order): `[16, −16, 0, 8]`
- kernels produce: `[8, 0, −16, 16]` = exactly the reversed order.

The golden test currently **locks the reversed (buggy) reality** so it can't
change silently; flip its assertion to `out == expected` once fixed.

### Open question for the user (blocks Wave 5)
Which layout is canonical?
- **(A)** Models load pre-packed from safetensors already in the compute kernels'
  MSB-first order ⇒ the compute kernels are correct and
  `execute_to_bitnet` (the f32→bitnet quantiser) is the buggy path (its
  "matches safetensors" comment would be wrong). Fix = flip the packer's shifts.
- **(B)** `execute_to_bitnet`'s LSB-first "matches safetensors" is correct ⇒ ALL
  four compute kernels read rows reversed. Fix = flip the compute shifts
  (`>>6`↔`>>0`, `>>4`↔`>>2`) in swar/avx2/sse/scalar.

Either fix is one-directional and small, but choosing WRONG silently permutes
every BitNet layer's output rows. Needs a ground-truth check against a real
pre-packed safetensors BitNet weight (or the model author's intent) — not a
guess. Until decided, bitnet migration is on hold; matmul and quantization (the
other Wave-5 items) are independent and can proceed first.
