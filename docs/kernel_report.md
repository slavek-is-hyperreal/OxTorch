# HPC Kernel Report - OxTorch CPU Backend (v3.8.1-rc)

This report tracks the optimization status of the native CPU kernels in the **v3.8.1-rc Unified Core**.

## 1. Dispatch Model
- ✅ **STATIC** - Compiled-in feature specialization.
- ⚡ **ASM-INLINED** - Manual assembly or SIMD intrinsics.
- ❌ **FALLBACK** - Generic Rust/Scalar implementation.

---

## 2. Binary Operations Logic

### Op: `add`
| Precision | SSE2 | AVX1 | AVX2 | NEON | Dispatch |
|---|---|---|---|---|---|
| BF16 | ✅ | ✅ | ❌ | ✅ | Static |
| F16  | ✅ | ✅ | ❌ | ✅ | Static |
| F32  | ⚡ | ⚡ | ✅ | ⚡ | Static |
| I8   | ❌ | ✅ | ✅ | ✅ | Static |

### Op: `sub`
| Precision | SSE2 | AVX1 | AVX2 | NEON | Dispatch |
|---|---|---|---|---|---|
| BF16 | ✅ | ✅ | ❌ | ✅ | **NEW** (v3.8.1) |
| F32  | ✅ | ✅ | ✅ | ✅ | **NEW** (v3.8.1) |
| I8   | ✅ | ✅ | ✅ | ✅ | Static |

### Op: `mul`
| Precision | SSE2 | AVX1 | AVX2 | NEON | Dispatch |
|---|---|---|---|---|---|
| BF16 | ✅ | ✅ | ❌ | ✅ | Static |
| F16  | ❌ | ✅ | ❌ | ❌ | **NEW** (v3.8.1) |
| F32  | ✅ | ✅ | ✅ | ✅ | Static |
| I8   | ✅ | ✅ | ✅ | ✅ | Static |

---

## 3. Reduction Operations Logic

| Op | BF16 | F16 | F32 | I8 | Tech |
|---|---|---|---|---|---|
| **Sum** | ✅ | ✅ | ✅ | ✅ | `f64` Accumulation |
| **Max** | ✅ | ✅ | ✅ | ✅ | SIMD `vmaxps` |
| **Mean** | ✅ | ✅ | ✅ | ✅ | `f64` Accumulation |

> **Note on f16/bf16 Summation**:
> OxTorch implements a **Drain Barrier** every 1024 elements in `sum_f16_f16c` to avoid floating-point drift. This ensures bit-perfect parity with PyTorch's reference implementation.
