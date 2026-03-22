# OxTorch vs. PyTorch — Gap Analysis

> **Cel**: pełna paryteta z PyTorch dla inferencji na każdej klasie sprzętu (PC/Server/Mobile).
> **Stan na**: 2026-03-22 | OxTorch v3.7.0

---

## 1. Co mamy teraz (OxTorch Core)

```text
Tensor API:
  shape, device, dtype, name                     — właściwości bazowe
  Tensor.from_ssd(path, shape, dtype)            — unikalny SSD streaming
  __add__, __sub__, __mul__, __matmul__          — operatory arytmetyczne native
  zeros(), ones(), rand()                        — fabryki (native/fallback)
  cat(), stack(), split(), chunk()               — operacje sekwencyjne
  relu, gelu, silu, tanh, exp                    — aktywacje SIMD
  layer_norm, rms_norm                           — normalizacje (Transformers)
  index_select / embedding                       — kluczowe dla LLM
Tryby: CPU | VULKAN | HYBRID | SSD streaming (MSTS)
Precyzja: F32 | F16 | BF16 | INT8 | Ternary (BitNet 1.58b)
```

---

## 2. Tier 1 — KRYTYCZNE dla inference

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

## 3. Tier 2 — WAŻNE dla LLM & Advanced Models

| Op | PyTorch | OxTorch | Status | Priorytet |
|:---|:---|:---|:---|:---|
| **Embeddings** | `F.embedding` | ✅ `index_select` | **DONE** | — |
| **Attention (SDPA)** | `scaled_dot_product_attention` | ❌ fallback | **TODO** | 🔴 Krytyczne LLM |
| **Conv2D / Conv1D** | `F.conv2d`, `F.conv1d` | ❌ fallback | **TODO** | 🟡 CNN |
| **Exp / Log / Sqrt** | `torch.exp` | ✅ partial | **PARTIAL** | 🟡 Precyzja |
| **Pow / Square** | `torch.pow` | ❌ fallback | **TODO** | 🟡 RMSNorm-fused |
| **Argmax / TopK** | `torch.argmax`, `torch.topk` | ❌ fallback | **TODO** | 🟡 Decoding |
| **Abs** | `torch.abs` | ❌ fallback | **TODO** | 🟢 Łatwe |
| **Indexing (Slicing)** | `x[0, :, 2]` | ❌ fallback | **TODO** | 🔴 UX |
| **Broadcast logic** | automatyczny | ❌ brak | **TODO** | 🔴 UX |

---

## 4. Tier 3 — ZAAWANSOWANE (Low Priority)

| Kategoria | Przykłady | Status |
|:---|:---|:---:|
| **LAPACK / Linalg** | `inv`, `cholesky`, `svd` | ❌ fallback |
| **Spektral (FFT)** | `torch.fft` | ❌ fallback |
| **Autograd** | `.backward()` | ❌ No-plan |
| **Advanced Indexing** | `gather`, `scatter_add` | ❌ fallback |

---

## 🚀 Przewaga OxTorch: Wykraczając poza PyTorch

| Funkcja | Opis | Stan |
|:---|:---|:---:|
| **MSTS SSD** | Natywna obsługa tensorów Out-of-RAM (100GB+). | **DONE** |
| **Ternary SIMD** | Kernele `{-1, 0, 1}` dla oszczędności 10x pamięci. | **DONE** |
| **Hybrid Race** | Równoległe nasycenie CPU + GPU na jednym opie. | **DONE** |
| **Tri-Precision** | Mix F32/F16/BF16 bez emulacji software'owej. | **DONE** |

---
*Gap Analysis jest dokumentem żywym. Każdy commit poprawiający natywne wsparcie musi być tu odnotowany.*
