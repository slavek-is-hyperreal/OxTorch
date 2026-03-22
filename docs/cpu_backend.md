# CPU Backend & SIMD Architecture

Backend CPU w OxTorch został zaprojektowany pod kątem wysokiej wydajności na architekturach x86_64 oraz aarch64, ze szczególnym uwzględnieniem maszyn bez instrukcji AVX-512 (np. Ivy Bridge, Haswell).

---

## 1. Struktura Folderów

Kernele CPU są ściśle uporządkowane według kategorii operacji i precyzji:

```text
vulkannn_rusted/src/cpu/ops/
├── binary/          # Operacje na dwóch tensorach (add, sub, mul, div)
├── unary/           # Operacje na jednym tensorze (relu, gelu, exp, sigmoid)
├── matmul/          # Mnożenie macierzy (f32, f16, bf16, i8)
├── indexing/        # index_select, embedding
├── reduction/       # sum, mean, max, softmax
├── sequence/        # cat, stack, split, chunk
└── norm/            # layer_norm, rms_norm
```

Każda operacja (np. `relu`) posiada własny folder, a w nim pliki dla konkretnych typów danych: `f32.rs`, `f16.rs`, `bf16.rs`, `i8.rs`.

---

## 2. Manual S.O.P: Dodawanie nowej funkcji

Aby dodać nową operację procesorową (np. `abs`), postępuj zgodnie z poniższą listą:

### Krok 1: Tworzenie struktury
Stwórz folder `src/cpu/ops/unary/abs/` i dodaj pliki `mod.rs`, `f32.rs`, `f16.rs`, `bf16.rs`, `i8.rs`. Zarejestruj moduł w `src/cpu/ops/unary/mod.rs`.

### Krok 2: Implementacja Kerneli (SIMD -> Fallback)
W każdym pliku implementuj funkcję z jawną specjalizacją. Przykład dla `f32.rs`:
- `abs_f32_avx()` – wykorzystujące `_mm256_andnot_ps` (maskowanie bitu znaku).
- `abs_f32_sse()` – dla starszych procesorów.
- `abs_f32_neon()` – dla ARM.
- `abs_f32_scalar()` – obowiązkowy fallback dla pozostałych architektur.

### Krok 3: Integracja z Python API (Odpięcie Fallbacku)
1.  Dodaj metodę `execute_abs` do `vulkannn_rusted/src/tensor/ops.rs`.
2.  W `vulkannn_rusted/oxtorch/tensor.py` dodaj metodę:
    ```python
    def abs(self):
        return self._vnn.execute_abs()
    ```
Dodanie tej metody do klasy `Tensor` w Pythonie automatycznie "nadpisuje" mechanizm `__getattr__`, tym samym odpinając wolny fallback PyTorcha.

### Krok 4: Testy i Parity
Stwórz benchmark w `tests/benchmarks/f32/abs_cpu.py` (dziedzicząc po `BenchmarkBase`). Uruchom go, aby sprawdzić "bit-perfect" parity z PyTorchem.

### Krok 5: Regresja Całościowa
**BARDZO WAŻNE**: Przed wysłaniem PR, uruchom **WSZYSTKIE** testy:
```bash
python tests/run_all_benchmarks.py
```
Należy upewnić się, że nowa implementacja nie wpłynęła na stabilność alokacji w `TensorPool` lub nie zepsuła orkiestracji MSTS.

---

## 3. Współpraca z MSTS (Wielowątkowość)

Kernele CPU w OxTorch **nie powinny** same w sobie używać wielowątkowości (np. wewnętrznej pętli `par_iter`). Są one orkiestrowane przez system MSTS na poziomie wyższym:

*   **Path A (Direct)**: Kernel jest wołany raz na całościowym buforze. Wykorzystuje 1 rdzeń (maksymalna kontrola cache).
*   **Path B (Single-mode)**: Kernel przetwarza kafelki 1MB sekwencyjnie. Dane idealnie mieszczą się w L2, zapewniając zero cache misses.
*   **Path C (Full Parallel)**: `MSTS` używa `Rayon` do wysłania różnych kafelek na różne rdzenie. Każdy rdzeń wykonuje **jednowątkowy** kernel na swoim kaflu. Dzięki temu unikamy walki o cache między wątkami (False Sharing).

---

## 4. Zasady implementacji SIMD

1.  **Alignment**: Zawsze zakładaj, że dane mogą nie być wyrównane do 32-bajtów (używaj `_mm256_loadu_ps` zamiast `load_ps`).
2.  **Tails**: Jeśli rozmiar tensora nie jest wielokrotnością szerokości rejestru (np. 8 dla AVX), zawsze obsłuż "ogon" pętlą skalarną lub maskowaną (`AVX-512`).
3.  **No Exceptions**: Kod kernela musi być `panic-free`.
