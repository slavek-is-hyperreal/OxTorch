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

### RESOLUTION (ground truth, 2026-07) — canonical = MSB-first ⇒ Option A
Source: Microsoft **BitNet** `src/ggml-bitnet-mad.cpp` (`quantize_i2_s` +
`ggml_vec_dot_i2_i8_s_1x1` / `_1x4` / `_1xN` / `_Nx1`, main branch,
https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-mad.cpp):
- Packer: `i2_weight[…] |= q8[…] << (6 - 2*group_idx)` ⇒ the FIRST element
  (`group_idx 0`) lands in bits **[7:6] (MSB)**; the last (`group_idx 3`) in
  **[1:0] (LSB)**.
- Kernel: `xq8_0 = (byte >> 6) & 0x03` (first element ← MSB). Packer and kernel
  agree with each other — MSB-first is the canonical convention this ecosystem
  (bitnet.cpp / GGUF I2_S) is built on.

Mapped onto OxTorch:
- OxTorch **compute kernels** (swar/avx2/sse/scalar) read row0 from `(byte>>6)&3`
  = **MSB-first ⇒ they MATCH the canonical convention and are CORRECT.**
- OxTorch **packer** `execute_to_bitnet` writes row0 into `(q0<<0)` = LSB ⇒ it is
  the OUTLIER. **This is Option A: fix the PACKER, not the kernels.**

**Recommended fix (deferred — see caveats):** in `tensor/conversion.rs::
execute_to_bitnet` (BitNet2), pack MSB-first:
`(q0<<6)|(q1<<4)|(q2<<2)|(q3<<0)` instead of `(q0<<0)|(q1<<2)|(q2<<4)|(q3<<6)`.
That makes OxTorch internally consistent AND aligned with bitnet.cpp/GGUF.

### Two caveats before APPLYING the fix (why it is not applied yet)
1. **Grouping is a separate axis, unverified.** Microsoft I2_S packs 4 *strided*
   elements per byte — `{p, p+32, p+64, p+96}` from a 128-element block
   (`group_pos = j % 32`, `group_idx = j / 32`) — NOT 4 consecutive weights.
   OxTorch's `execute_to_bitnet` packs 4 *consecutive rows* per byte
   (row-interleaved 1×4). So OxTorch's byte-*grouping* is its own format, not
   GGUF I2_S. Pack and compute in OxTorch AGREE on the grouping (both use
   4-consecutive-rows), so this does not cause an internal bug — but it means
   OxTorch weights are NOT byte-compatible with GGUF I2_S, only with themselves.
2. **`.safetensors` provenance unconfirmed.** The packer's "matches safetensors"
   comment points at the HF `microsoft/bitnet-b1.58-2B-4T` (non-GGUF) format. The
   GGUF I2_S source above is *very strong corroborating evidence* for MSB-first,
   but does not by itself prove the byte order inside a real HF `.safetensors`
   produced by a different converter. Before flipping the packer, byte-inspect a
   real pre-packed HF tensor (or the HF repo's packing code) to confirm MSB-first
   there too.

### Consequence for Wave 5
- The **compute kernels are established CORRECT** (MSB-first), so **bitnet
  compute may migrate as move-not-rewrite** without change.
- The **packer fix (Option A) is deferred** pending the real-safetensors byte
  check (caveat 2). It is a `tensor/conversion.rs` change, separable from the
  cpu-kernel migration. Flagged, not silently dropped.
- **matmul and quantization are independent** and proceed first (user directive).
