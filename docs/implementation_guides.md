# Implementation Guides & Phase 2 Roadmap

This document serves as a dynamic guide to the current implementation and a template for new functionalities. Core architecture information has been moved to dedicated pages (see: `README.md`).

---

## 1. Implementation Template (New Op S.O.P)

When introducing a new operation (Phase 2 and beyond), fill out the following template in a new guide file:

### [OPERATION NAME]
- **Target**: (e.g., F32, F16, BF16, I8)
- **Status**: 📝 TODO / 🚧 WIP / ✅ DONE
- **CPU Kernels**: [ ] f32 [ ] f16 [ ] bf16 [ ] i8
- **Vulkan Kernels**: [ ] shader [ ] pipeline [ ] backend dispatch
- **MSTS Support**: [ ] Path A [ ] Path B [ ] Path C
- **Python Integration**: [ ] `oxtorch/tensor.py` [ ] `oxtorch/__init__.py`

---

## 2. Phase 2: Embeddings & Indexing (CURRENT)

*The goal is to natively replace PyTorch Fallback for loading language models (Embedding Layer).*

### Vectorization Strategy (HPC Research):
1.  **Cache Prefetching**: Due to scattered loads, we use `_mm_prefetch(..., _MM_HINT_T0)` for weight rows.
2.  **AVX-512 Gather**: Utilizing `_mm512_i32gather_ps` for index jumping where possible.
3.  **Vulkan Dispatch**: Mapping `[N, F]` -> `gl_GlobalInvocationID`. We organize GPU threads to read the same index address from L1.

---

## 3. Sprint Archive (Completed)

Technical descriptions of the following functionalities can be found in:
-   **MSTS 3-Path Dispatch**: `docs/msts_logic.md`
-   **Execution Modes & Hybrid**: `docs/execution_modes.md`
-   **SIMD & CPU Folders**: `docs/cpu_backend.md`
-   **Python Fallback Mechanism**: `docs/oxtorch_package.md`

| Sprint | Functionality | Status |
|:---|:---|:---:|
| 1 | Elementwise (Vulkan + CPU) | ✅ DONE |
| 1 | Softmax | ✅ DONE |
| 1 | BitNet 1.58b Linear | ✅ DONE |
| 4 | TensorPool Slab Allocator | ✅ DONE |
| 4 | MSTS Compile-time Burn-in | ✅ DONE |
