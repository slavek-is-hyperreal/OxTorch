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
*Status: OxTorch v3.8.0 Ready*
