# OxTorch SIMD Support Matrix

These tables represent the level of SIMD instruction support for individual operations in OxTorch.
- **✅** : Dedicated, optimized SIMD kernel.
- **❌** : No dedicated kernel (uses scalar fallback or general-purpose registers).
- **(upcast/approx)** : Operation performed by casting to a higher precision (e.g., F32) or bitwise approximation.

---

## 1. F32 Precision (Float 32)

| Function | no_avx (GPR) | avx1 | avx2 | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MatMul / Linear** | ✅ SGEMM | ✅ SGEMM | ✅ SGEMM | ✅ SGEMM | ✅ SGEMM | ✅ SGEMM |
| **Add / Sub / Mul / Div** | ✅ SSE2 | ✅ Native | ✅ Native | AVX2* | ✅ Scalar | ✅ Native |
| **ReLU** | ✅ Scalar | ✅ Native | ✅ Native | AVX2* | ✅ Scalar | ✅ Native |
| **GeLU / Sigmoid / SiLU** | ✅ SSE2 | ✅ Native | ✅ Native | AVX2* | ✅ Scalar | ✅ Native |
| **Softmax** | ✅ Scalar | ✅ Scalar | ✅ Scalar | ✅ Scalar | ✅ Scalar | ✅ Scalar |
| **RMSNorm / LayerNorm** | ✅ Scalar | ✅ Native | ✅ Native | AVX2* | ✅ Scalar | ✅ Scalar |
| **IndexSelect / Embedding** | ✅ Scalar | ✅ Native | ✅ Native | ✅ Native | ✅ Scalar | ✅ Native |

*\* No dedicated AVX-512 instructions; falls back to the AVX2 path.*

---

## 2. F16 Precision (Half Precision)

| Function | no_avx (GPR) | avx1 (F16C) | avx2 (F16C) | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MatMul (Tiled 256x256)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Add / Sub / Mul / Div** | ✅ Scalar | ✅ Upcast | ✅ Upcast | Upcast | ✅ Scalar | ✅ Native |
| **ReLU** | ✅ Scalar | ✅ Upcast | ✅ Upcast | Upcast | ✅ Scalar | ✅ Native |
| **IndexSelect / Embedding** | ✅ SWAR | ✅ Native | ✅ Native | ✅ Native | ✅ Scalar | ✅ Native |

---

## 3. BF16 Precision (Brain Float 16)

| Function | no_avx (GPR) | avx1 | avx2 | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MatMul (Tiled 256x256)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Add / Sub / Mul / Div** | ✅ Scalar | ✅ Upcast | ✅ Upcast | Upcast | ✅ Scalar | ✅ Scalar |
| **ReLU** | ✅ Scalar | ✅ Upcast | ✅ Upcast | Upcast | ✅ Scalar | ✅ Scalar |
| **IndexSelect / Embedding** | ✅ SWAR | ✅ Native | ✅ Native | ✅ Native | ✅ Scalar | ✅ Native |

---

## 4. INT8 Precision (Quantized)

| Function | no_avx | avx1 | avx2 | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Binary Add / Sub** | ✅ SWAR* | ✅ Scalar | ✅ Native | AVX2* | ✅ Scalar | ✅ Native |
| **ReLU** | ✅ Scalar | ✅ Scalar | ✅ Native | AVX2* | ✅ Scalar | ✅ Scalar |
| **Sigmoid / GeLU** | ✅ LUT | ✅ LUT | ✅ LUT | ✅ LUT | ✅ LUT | ✅ LUT |
| **Max Reduction** | ✅ Scalar | ✅ Native | ✅ Native | AVX2* | ✅ Scalar | ✅ Scalar |
| **IndexSelect** | ✅ SWAR | ✅ Native | ✅ Native | ✅ Native | ✅ Scalar | ✅ Native |

*\* SWAR in `add_i8` now has saturating fallback detection.*

---

## Key Findings (Post-Phase 1.6):

1.  **NEON Integration**: ARM support gaps have been significantly reduced. `ReLU`, `GELU`, `Exp`, `Sum/Max`, and unary functions now have native NEON paths.
2.  **Transcendental SIMD**: Sigmoid, SiLU, and Tanh are fully vectorized (AVX2/NEON) using high-quality polynomial approximations.
3.  **Memory Efficiency**: `TensorPool` was introduced, eliminating allocation overhead in `LayerNorm`, `RMSNorm`, and tiled matrix conversion in `MatMul`.
4.  **MatMul Tiling**: F16/BF16 support no longer requires allocating entire F32 matrices, enabling the execution of large models (LLMs) on memory-constrained systems.
