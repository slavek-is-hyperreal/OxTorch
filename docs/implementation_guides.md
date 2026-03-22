# Implementation Guides & Phase 2 Roadmap

Ten dokument służy jako dynamiczny przewodnik po bieżącej implementacji i szablon dla nowych funkcjonalności. Informacje o architekturze rdzenia zostały przeniesione do dedykowanych stron (patrz: `README.md`).

---

## 1. Szablon Implementacji (New Op S.O.P)

Przy wprowadzaniu nowej operacji (Faza 2 i dalej) należy wypełnić poniższy szablon w nowym pliku guide:

### [NAZWA OPERACJI]
- **Target**: (np. F32, F16, BF16, I8)
- **Status**: 📝 TODO / 🚧 WIP / ✅ DONE
- **Kernele CPU**: [ ] f32 [ ] f16 [ ] bf16 [ ] i8
- **Kernele Vulkan**: [ ] shader [ ] pipeline [ ] backend dispatch
- **MSTS Support**: [ ] Path A [ ] Path B [ ] Path C
- **Python Integration**: [ ] `oxtorch/tensor.py` [ ] `oxtorch/__init__.py`

---

## 2. Faza 2: Embeddings & Indexing (BIEŻĄCE)

*Celem jest natywne zastąpienie PyTorch Fallback dla wczytywania modeli językowych (Emb Layer).*

### Strategia wektoryzacji (HPC Research):
1.  **Cache Prefetching**: Z uwagi na scattered loads, używamy `_mm_prefetch(..., _MM_HINT_T0)` dla wierszy wagi.
2.  **AVX-512 Gather**: Wykorzystanie `_mm512_i32gather_ps` dla skoku w indeksach tam, gdzie to możliwe.
3.  **Vulkan Dispatch**: Mapowanie `[N, F]` -> `gl_GlobalInvocationID`. Wątki GPU organizujemy tak, by czytały ten sam adres indeksu z L1.

---

## 3. Archiwum Sprintów (Wykonano)

Szczegółowe opisy techniczne poniższych funkcjonalności znajdują się w:
-   **MSTS 3-Path Dispatch**: `docs/msts_logic.md`
-   **Execution Modes & Hybrid**: `docs/execution_modes.md`
-   **SIMD & CPU Folders**: `docs/cpu_backend.md`
-   **Python Fallback Mechanism**: `docs/oxtorch_package.md`

| Sprint | Funkcjonalność | Status |
|:---|:---|:---:|
| 1 | Elementwise (Vulkan + CPU) | ✅ DONE |
| 1 | Softmax | ✅ DONE |
| 1 | BitNet 1.58b Linear | ✅ DONE |
| 4 | TensorPool Slab Allocator | ✅ DONE |
| 4 | MSTS Compile-time Burn-in | ✅ DONE |
