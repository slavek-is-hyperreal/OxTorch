# OxTorch vs. PyTorch — Gap Analysis

> **Goal**: Full parity with PyTorch for inference on every hardware class (PC/Server/Mobile).
> **Status as of**: 2026-03-22 | OxTorch v3.7.0

---

## 1. Current Status (OxTorch Core)

```text
Tensor API:
  shape, device, dtype, name                     — Base properties
  Tensor.from_ssd(path, shape, dtype)            — Unique SSD streaming
  __add__, __sub__, __mul__, __matmul__          — Native arithmetic operators
  zeros(), ones(), rand()                        — Factories (Native/Fallback)
  cat(), stack(), split(), chunk()               — Sequential operations
  relu, gelu, silu, tanh, exp                    — SIMD activations
  layer_norm, rms_norm                           — Normalizations (Transformers)
  index_select / embedding                       — Critical for LLMs
Execution Modes: CPU | VULKAN | HYBRID | SSD streaming (MSTS)
Precision: F32 | F16 | BF16 | INT8 | Ternary (BitNet 1.58b)
```

---

## 2. Tier 1 — CRITICAL for Inference

| Op | PyTorch | OxTorch | Status |
|:---|:---|:---|:---:|
| **MatMul 2D / Batch** | `torch.mm`, `torch.bmm` | ✅ `bmm()` | **DONE** |
| **Linear Layer** | `F.linear(x, W, b)` | ✅ native + bias | **DONE** |
| **Elementwise Ops** | `+`, `-`, `*`, `/` | ✅ native SIMD | **DONE** |
| **Activations** | `ReLU`, `GELU`, `SiLU`, `Tanh` | ✅ native SIMD | **DONE** |
| **Normalizations** | `LayerNorm`, `RMSNorm` | ✅ native | **DONE** |
| **Reductions** | `sum`, `mean`, `max`, `min` | ✅ native | **DONE** |
| **Reshape / View** | `.view()`, `.reshape()` | ✅ native | **DONE** |
| **Squeeze / Unsqueeze** | `.squeeze()`, `.unsqueeze()` | ✅ native | **DONE** |
| **Concatenate / Stack** | `torch.cat`, `torch.stack` | ✅ native | **DONE** |
| **Split / Chunk** | `torch.split`, `torch.chunk` | ✅ native | **DONE** |
| **Scalar Ops** | `x * 2.0`, `x + 1` | ✅ native | **DONE** |
| **Creators** | `zeros`, `ones`, `rand` | ✅ native | **DONE** |

---

## 3. Tier 2 — IMPORTANT for LLMs & Advanced Models

| Op | PyTorch | OxTorch | Status | Priority |
|:---|:---|:---|:---|:---|
| **Embeddings** | `F.embedding` | ✅ `index_select` | **DONE** | — |
| **Attention (SDPA)** | `scaled_dot_product_attention` | ❌ fallback | **TODO** | 🔴 Critical LLM |
| **Conv2D / Conv1D** | `F.conv2d`, `F.conv1d` | ❌ fallback | **TODO** | 🟡 CNN |
| **Exp / Log / Sqrt** | `torch.exp` | ✅ partial | **PARTIAL** | 🟡 Precision |
| **Pow / Square** | `torch.pow` | ❌ fallback | **TODO** | 🟡 RMSNorm-fused |
| **Argmax / TopK** | `torch.argmax`, `torch.topk` | ❌ fallback | **TODO** | 🟡 Decoding |
| **Abs** | `torch.abs` | ❌ fallback | **TODO** | 🟢 Easy |
| **Indexing (Slicing)** | `x[0, :, 2]` | ❌ fallback | **TODO** | 🔴 UX |
| **Broadcast logic** | automatic | ❌ missing | **TODO** | 🔴 UX |

---

## 4. Tier 3 — ADVANCED (Low Priority)

| Category | Examples | Status |
|:---|:---|:---:|
| **LAPACK / Linalg** | `inv`, `cholesky`, `svd` | ❌ fallback |
| **Spectral (FFT)** | `torch.fft` | ❌ fallback |
| **Autograd** | `.backward()` | ❌ No-plan |
| **Advanced Indexing** | `gather`, `scatter_add` | ❌ fallback |

---

## 🚀 The OxTorch Advantage: Beyond PyTorch

| Feature | Description | Status |
|:---|:---|:---:|
| **MSTS SSD** | Native support for Out-of-RAM tensors (100GB+). | **DONE** |
| **Ternary SIMD** | `{-1, 0, 1}` kernels for 10x memory savings. | **DONE** |
| **Hybrid Race** | Parallel CPU + GPU saturation on a single op. | **DONE** |
| **Tri-Precision** | Mix F32/F16/BF16 without software emulation. | **DONE** |

---
*Gap Analysis is a living document. Every commit improving native support must be recorded here.*
