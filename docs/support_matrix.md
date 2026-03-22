# OxTorch SIMD Support Matrix

Te tabele przedstawiają poziom wsparcia instrukcji SIMD dla poszczególnych operacji w OxTorch. 
- **✅** : Dedykowany, zoptymalizowany kernel SIMD.
- **❌** : Brak dedykowanego kernela (używany fallback skalarny lub ogólny GPR).
- **(upcast/approx)** : Operacja wykonywana przez rzutowanie na wyższą precyzję (np. F32) lub aproksymację bitową.

---

## 1. Precyzja F32 (Float 32)

| Funkcja | no_avx (GPR) | avx1 | avx2 | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MatMul / Linear** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Add / Sub / Mul / Div** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **ReLU** | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **GeLU / Sigmoid / SiLU** | ✅ | ✅ (approx) | ❌ | ❌ | ✅ | ❌ |
| **Softmax** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **RMSNorm / LayerNorm** | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **IndexSelect / Embedding** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 2. Precyzja F16 (Half Float)

| Funkcja | no_avx (GPR) | avx1 (F16C) | avx2 (F16C) | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MatMul (upcast F32)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Add / Sub / Mul / Div** | ✅ | ✅ (F16C) | ✅ (F16C) | ❌ | ✅ | ❌ |
| **ReLU** | ✅ | ✅ (F16C) | ✅ (F16C) | ❌ | ✅ | ❌ |
| **IndexSelect / Embedding** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 3. Precyzja BF16 (Brain Float)

| Funkcja | no_avx (GPR) | avx1 | avx2 | avx512 | arm64 | NEON |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MatMul (upcast F32)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
## VulkanNN CPU SIMD Support Matrix

Oto szczegółowa macierz wsparcia dla instrukcji SIMD w backendzie CPU, opracowana na podstawie audytu kodu źródłowego (`vulkannn_rusted/src/cpu`).

## Typy wsparcia:
- **Native SIMD**: Dedykowany kernel wykorzystujący natywne instrukcje dla danej precyzji (np. `_mm256_add_ps` dla F32).
- **SIMD (Upcast)**: Kernel wykorzystujący SIMD po uprzedniej konwersji do F32 (częste dla F16/BF16 na starszych x86).
- **SWAR**: *SIMD Within A Register* - optymalizacja dla procesorów bez SIMD przy użyciu rejestrów GP (64-bit).
- **Scalar**: Brak wektoryzacji, operacje wykonywane element po elemencie.

---

## 1. Precyzja: F32 (Single Precision)

| Funkcja | no_avx | avx1 | avx2 | avx512 | arm32 | arm64 | NEON |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Binary (Add/Sub/Mul/Div)** | SSE2 | Native | Native | AVX2* | Scalar | Scalar | Native |
| **Scalar (Add/Sub/...)** | SSE2 | Native | Native | AVX2* | Scalar | Scalar | Scalar |
| **ReLU** | Scalar | Native | Native | AVX2* | Scalar | Scalar | Native |
| **GELU** | SSE2 | Native | Native | AVX2* | Scalar | Scalar | Native |
| **Exp** | SSE2 | Native | Native | AVX2* | Scalar | Scalar | Native |
| **Sigmoid / SiLU / Tanh** | Scalar | Scalar | Native | Native* | Scalar | Scalar | Native |
| **Sum / Max Reduction** | Scalar | Native | Native | Native | Scalar | Scalar | Native |
| **Softmax** | Scalar | Scalar | Scalar | Scalar | Scalar | Scalar | Scalar |
| **Norm (Layer/RMS)** | Scalar | Native | Native | AVX2* | Scalar | Scalar | Scalar |
| **IndexSelect** | Scalar | Native | Native | Native | Scalar | Scalar | Native |
| **MatMul** | SGEMM | SGEMM | SGEMM | SGEMM | SGEMM | SGEMM | SGEMM |

*\* Brak dedykowanych instrukcji AVX-512, fallback do ścieżki AVX2.*

---

## 2. Precyzja: F16 (Half Precision)

| Funkcja | no_avx | avx1 | avx2 | avx512 | arm32 | arm64 | NEON |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Binary Add** | Scalar | Upcast | Upcast | Upcast | Scalar | Scalar | Native |
| **ReLU** | Scalar | Upcast | Upcast | Upcast | Scalar | Scalar | Native* |
| **Sum Reduction** | Scalar | Upcast | Upcast | Upcast | Scalar | Scalar | Native* |
| **Norm (Layer/RMS)** | Scalar | Upcast* | Upcast* | Upcast* | Scalar | Scalar | Scalar |
| **IndexSelect** | SWAR | Native | Native | Native | Scalar | Scalar | Native |
| **MatMul** | Tiled* | Tiled* | Tiled* | Tiled* | Tiled* | Tiled* | Tiled* |

*\* Operacje Zero-Allocation z użyciem `TensorPool`. MatMul wykorzystuje kafelkowanie 256x256.*

---

## 3. Precyzja: BF16 (Brain Float 16)

| Funkcja | no_avx | avx1 | avx2 | avx512 | arm32 | arm64 | NEON |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Binary Add** | Scalar | Upcast | Upcast | Upcast | Scalar | Scalar | Scalar |
| **ReLU** | Scalar | Upcast | Upcast | Upcast | Scalar | Scalar | Scalar |
| **Norm (Layer/RMS)** | Scalar | Upcast* | Upcast* | Upcast* | Scalar | Scalar | Scalar |
| **IndexSelect** | SWAR | Native | Native | Native | Scalar | Scalar | Native |
| **MatMul** | Tiled* | Tiled* | Tiled* | Tiled* | Tiled* | Tiled* | Tiled* |

*\* Operacje Zero-Allocation z użyciem `TensorPool`. MatMul wykorzystuje kafelkowanie 256x256.*

---

## 4. Precyzja: INT8 (Quantized)

| Funkcja | no_avx | avx1 | avx2 | avx512 | arm32 | arm64 | NEON |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Binary Add / Sub** | SWAR* | Scalar | Native | AVX2* | Scalar | Scalar | Native |
| **ReLU** | Scalar | Scalar | Native | AVX2* | Scalar | Scalar | Scalar |
| **Sigmoid** | LUT | LUT | LUT | LUT | LUT | LUT | LUT |
| **Max Reduction** | Scalar | Native | Native | AVX2* | Scalar | Scalar | Scalar |
| **IndexSelect** | SWAR | Native | Native | Native | Scalar | Scalar | Native |

*\* SWAR w `add_i8` posiada teraz bezpieczną detekcję nasycenia (saturating fallback).*

---

## Kluczowe wnioski (Post-Phase 1.6):

1.  **NEON Integration**: Znacznie zredukowano luki w obsłudze ARM. `ReLU`, `GELU`, `Exp`, `Sum/Max` oraz funkcje unarne posiadają teraz natywne ścieżki NEON.
2.  **Transcendental SIMD**: Sigmoid, SiLU i Tanh zostały w pełni zwektoryzowane (AVX2/NEON) z użyciem wysokiej jakości przybliżeń wielomianowych.
3.  **Memory Efficiency**: Wprowadzono `TensorPool`, co wyeliminowało narzut alokacji w `LayerNorm`, `RMSNorm` oraz przy kafelkowej konwersji macierzy w `MatMul`.
4.  **MatMul Tiling**: Obsługa F16/BF16 nie wymaga już alokacji całych macierzy F32, co pozwala na uruchamianie dużych modeli (LLM) przy ograniczonych zasobach RAM.
