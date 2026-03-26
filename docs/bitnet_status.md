# BitNet Inference Status Report (v3.7.1)

This document provides a technical overview of the current progress on BitNet-2B/3B native inference in OxTorch.

## Implementation Status Overview

| Component | Status | Details |
|:---|:---:|:---|
| **GGUF Loader** | ✅ **Complete** | Supports `I2_S` ternary weight parsing directly from Microsoft GGUF models. |
| **Modular Logic** | ✅ **Complete** | Refactored from monolithic `bit_linear.rs` to modular `src/cpu/ops/bitnet/` dispatcher. |
| **Tiered Kernels** | ✅ **Complete** | Specialized kernels implemented for **SSSE3**, **AVX2**, **AVX512**, and **SWAR**. |
| **Inference Loop** | ✅ **Functional** | End-to-end token generation loop is stable and no longer hangs. |
| **Numerical Parity** | ❌ **Broken** | Currently producing "garbage output". |

---

## 🛠️ What Works

### 1. Tiered SIMD Dispatcher
The engine now correctly detects hardware capabilities at runtime and dispatches to the most efficient kernel. For the current target hardware (IVY Bridge/Sandy Bridge), it uses the **128-bit SSSE3** path (`_mm_maddubs_epi16`), which is significantly faster than the scalar fallback.

### 2. GGUF I2_S Parsing
We have successfully mapped the Microsoft-specific `I2_S` quantization layout.
-   **Ternary Offset Correction**: GGUF weights stored as `{1, 2, 3}` are correctly remapped to `{0, 1, 2}` to align with our kernel logic (where `-1` is subtracted internally).
-   **Dynamic Data Offsets**: The loader now dynamically calculates the offset for ternary weights and logical scales, preventing memory corruption.

### 3. Stability
Generation no longer hangs. The model processes tokens at approximately **0.65 - 1.2 tok/s** (on CPU), which is a massive improvement over the previous unoptimized state.

---

## 🛑 What Doesn't Work (Current Blockers)

### 1. The Garbage/Nonsense Issue
While the model generates tokens, the output is nonsense (e.g., `vořínunvořínun...`). This is a "Numerical Divergence" issue.

**Suspected Causes:**
-   **Weight Swizzle Mismatch**: The official BitNet kernels in `ggml-bitnet-mad.cpp` expect a specific bit-swizzled layout within each 32-byte block. Our current loader attempts to unpack them into a linear format, but this process may be misaligned with how the SIMD kernels read them.
-   **Activation Normalization**: BitNet requires a specific `RMSNorm` configuration (usually with a different eps or sub-normalization step) that might be slightly out of sync with our current implementation.
-   **RoPE (Rotary Positional Embeddings)**: Any slight misalignment in RoPE precomputation or application results in total output collapse.

---

## 📅 Next Steps

1.  **Parity Audit of `load_bitlinear`**: Compare our Rust unpacking logic line-by-line with the `quantize_i2_s` and `preprocess_weights` functions in the official repository.
2.  **SSE Kernel Optimization**: Focus on the SSSE3 path to match the exact bit-layout of the 3B model's weights without unnecessary re-packing.
3.  **Numerical Checkpointing**: Print activations after the first LayerNorm and first self-attention layer to compare against the official implementation.

> [!TIP]
> **Honest Assessment**: We have achieved the "Hardware Plumbing" goal (it runs fast and stable), but the "Scientific Parity" (outputting correct words) is still 1-2 debugging cycles away.
