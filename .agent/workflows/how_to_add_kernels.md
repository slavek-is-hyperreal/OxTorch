---
description: How to add new kernels
---

# 📜 Konstytucja OxTorch: Protokół Dodawania Jader CPU

Niniejszy dokument definiuje standardy inżynieryjne "Scientific-Grade" dla implementacji nowych operacji matematycznych w backendzie CPU OxTorch.

## 1. Nazewnictwo i Lokalizacja
Każde jądro musi znajdować się w ściśle określonej strukturze:
`vulkannn_rusted/src/cpu/ops/[category]/[op]/[dtype]/`

**Wzór pliku**: `[op]_[dtype]_[arch].rs`
- Przykład: `sub_f32_avx1.rs`
- Przykład: `mul_bf16_avx2.rs`

## 2. Architektura Rejestracji (4 Stopnie)

### Stopień I: Lokalna Specjalizacja
Plik `[op]_[dtype]_[arch].rs` zawiera czyste jądro (często z użyciem intrinsics) zoptymalizowane pod konkretny zestaw instrukcji.

### Stopień II: Moduł Typu Danych (Compile-Time Dispatch)
Plik `mod.rs` wewnątrz folderu `[dtype]/`. 
Eksponuje funkcję `pub fn [op](...)`. Używa makr `#[cfg(target_feature = "...")]` do wyboru najszybszego dostępnego jądra w czasie kompilacji.
```rust
pub fn sub(a: &[f32], b: &[f32], res: &mut [f32]) {
    #[cfg(target_feature = "avx")] {
        return avx1::sub(a, b, res);
    }
    scalar::sub(a, b, res)
}
```

### Stopień III: Brama Równoległa (Rayon)
Plik `mod.rs` w folderze `[op]/`. 
Implementuje wielowątkowość. Dzieli wektory na mniejsze kawałki (`par_chunks_mut`) i przekazuje je do modułu typu danych.

### Stopień IV: Dispatcher MSTS
Plik `vulkannn_rusted/src/tensor/msts.rs`.
Rejestruje operację w systemie OxTorch, kierując ją na ścieżkę **RAM-FastPath**.

## 3. Protokół Wdrożenia (Workflow)
1. **Research**: Gemini Deep Research musi dostarczyć specyfikację optymalizacji (przepustowość portów, blokowanie rejestrów).
2. **Implementacja**: Kod musi być wyrównany do 64 bajtów (zgodność z `TensorPool`).
3. **Weryfikacja**:
   - Testy jednostkowe (Rust `#[test]`) dla krawędziowych przypadków.
   - Benchmarki Python (parity check z PyTorch).
4. **Sync**: Git Plumbing Merge po uzyskaniu 100% stabilności.

---

## 5. Konwencje ustalone w migracji cpu_old→cpu (precedensy — nie rozstrzygaj od nowa)

### 5.1 Szczebel i8 bez AVX2 = SSE4.1, NIE nowy wariant enuma `Arch`
`_mm_max_epi8` / signed-byte SIMD to **SSE4.1**, nie SSE2 (SSE2 nie ma
signed-byte max). Rozwiązanie przyjęte dla `relu_i8`: osobny plik
`[op]_i8_sse41.rs` dispatchowany **bezpośrednim** `is_x86_feature_detected!("sse4.1")`
w `i8/mod.rs` — NIE przez `active_arch()`, NIE jako `#[cfg]` wewnątrz pliku sse2,
NIE jako nowy wariant enuma `Arch`. Enum `Arch` (Scalar/Swar/Sse2/Avx1/Avx2/
Avx512/Neon) pozostaje nietknięty. Kolejne opy i8 mają iść tym samym wzorcem.

### 5.2 SWAR wraca do macierzy tylko ze zweryfikowanego źródła
Bit-trick SWAR (GPR-only fallback) wolno dodać **wyłącznie** gdy pochodzi z
autorytatywnego źródła (np. Hacker's Delight z numerem strony), NIE z wyobraźni,
i z testem różnicowym vs scalar. Dla i8 (256×256 kombinacji) test jest wyczerpujący
— użyj tego. Powód: `add_i8_swar` w legacy miał bug carry-leak między bajtami
(patrz `cpu/ops/binary/add/i8/mod.rs`). `TODO` z tą adnotacją > kernel z wiarą.

### 5.3 Transcendentalne (exp/sigmoid/silu/tanh/gelu): tolerancja i wyrocznia
Reguła 1 (transkrybuj legacy) jest **nadpisana** dla tej rodziny — współczynniki
z Cephes, gate = parity-vs-oracle (f64→f32) w granicy ULP, NIE vs legacy.
Pełna polityka + granice ULP + decyzja gelu=tanh-approx: `docs/kernel_specs/README.md`.
Każdy op ma `docs/kernel_specs/{op}_spec.md`.

### 5.4 Rozbieżności vs torch (div/0→0, relu(NaN)→0, …)
Nie „naprawiaj" ich w migracji (Reguła 6). Dopisz do `docs/known_divergences.md`
z decyzją KEEP/FIX/OPEN. Jeden świadomy przegląd po Fali 6.

### 5.5 Dwie powierzchnie unary
Każdy op unary eksponuje out-of-place `[op]` ORAZ in-place `[op]_inplace` na
Tier II i Tier III (msts woła in-place). Wzorzec: makro `tier3_unary!` w
`unary/relu/mod.rs`.

---
*Status: OxTorch v3.8.0 Ready (konwencje migracyjne §5 dodane 2026-07)*
