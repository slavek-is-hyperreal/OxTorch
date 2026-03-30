# OxTorch vs. PyTorch — Gap Analysis (v3.8.1-rc)

> **Goal**: Full parity with PyTorch for inference on every hardware class (PC/Server/Mobile).
> **Status as of**: 2026-03-30 | **OxTorch v3.8.1-rc**

---

## 🚀 The OxTorch Advantage: v3.8.1 Gains

| Feature | Description | Status |
|:---|:---|:---:|
| **MSTS v2 SSD** | 50% RAM Capacitor + io_uring zero-copy DMA. | **DONE** |
| **SIMD Core** | Optimized AVX1/NEON kernels for BF16/F16. | **DONE** |
| **Numerical Parity** | `f64` accumulation in reductions (sum, mean). | **DONE** |
| **BF16 Heroics** | **26x faster** Linear layer than PT on non-AVX512 CPUs. | **DONE** |

---

## 1. Tier 1 — CRITICAL for Inference

| Op | PyTorch | OxTorch | Status |
|:---|:---|:---|:---:|
| **Linear Layer** | `F.linear(x, W, b)` | ✅ SIMD + Bias | **DONE** |
| **Elementwise Ops** | `add`, `sub`, `mul`, `div` | ✅ SIMD Specializations | **DONE** |
| **Activations** | `ReLU`, `GELU`, `SiLU`, `Tanh` | ✅ Native SIMD | **DONE** |
| **Normalizations** | `LayerNorm`, `RMSNorm` | ✅ SIMD | **DONE** |
| **Reductions** | `sum`, `mean`, `max`, `min` | ✅ Parity Verified | **DONE** |
| **Sequential Ops** | `cat`, `stack`, `split`, `chunk` | ✅ Native | **DONE** |
| **Creators** | `zeros`, `ones`, `rand` | ✅ Native | **DONE** |

---

## 2. Tier 2 — IMPORTANT for LLMs & Advanced Models

| Op | PyTorch | OxTorch | Status | Priority |
|:---|:---|:---|:---|:---|
| **Embeddings** | `F.embedding` | ✅ `index_select` | **DONE** | — |
| **Attention (SDPA)** | `scaled_dot_product_attention` | ❌ fallback | **TODO** | 🔴 Critical LLM |
| **Conv2D / Conv1D** | `F.conv2d`, `F.conv1d` | ❌ fallback | **TODO** | 🟡 CNN |
| **Exp / Log / Sqrt** | `torch.exp` | ✅ partial | **PARTIAL** | 🟡 Precision |
| **Pow / Square** | `torch.pow` | ✅ SIMD | **DONE** | 🟡 RMSNorm-fused |
| **Argmax / TopK** | `torch.argmax`, `torch.topk` | ❌ fallback | **TODO** | 🟡 Decoding |
| **Abs** | `torch.abs` | ✅ SIMD | **DONE** | 🟢 Easy |
| **Indexing (Slicing)** | `x[0, :, 2]` | ❌ fallback | **TODO** | 🔴 UX |
| **Broadcast logic** | automatic | ❌ missing | **TODO** | 🔴 UX |

---

## 3. Tier 3 — ADVANCED (Low Priority)

| Category | Examples | Status |
|:---|:---|:---:|
| **LAPACK / Linalg** | `inv`, `cholesky`, `svd` | ❌ fallback |
| **Spectral (FFT)** | `torch.fft` | ❌ fallback |
| **Autograd** | `.backward()` | ❌ No-plan |
| **Advanced Indexing** | `gather`, `scatter_add` | ❌ fallback |

---

*Gap Analysis is a living document. Every commit improving native support must be recorded here.*
