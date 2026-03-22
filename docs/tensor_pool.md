# TensorPool: Slab Allocator dla Hot-Path

`TensorPool` to wysokowydajny, **Thread-Local Slab Allocator**, zaprojektowany w celu eliminacji systemowych alokacji (`malloc`/`free`) w gorących pętlach obliczeniowych OxTorch.

## 1. Dlaczego TensorPool?
Wiele operacji głębokiego uczenia (np. LayerNorm w F16) wymaga tymczasowych buforów na dane pośrednie (np. konwersja do F32). Wykonywanie alokacji `Vec::with_capacity` dla każdego wiersza tensora (row-wise) powoduje gigantyczny narzut i fragmentację pamięci. `TensorPool` rozwiązuje ten problem poprzez recykling buforów.

## 2. Architektura
`TensorPool` jest zaimplementowany jako struktura `ThreadLocal`, co oznacza, że każdy wątek roboczy (np. w Rayon) posiada własną pulę i nie konkuruje o blokady (lock-free access).

### System Kubłów (Bucketing)
Pula zarządza 6 pre-alokowanymi kubłami o różnych rozmiarach:
- **Tiny**: < 4 KB
- **Small**: < 64 KB
- **Medium**: < 1 MB
- **Large**: < 16 MB
- **X-Large**: < 256 MB
- **Massive**: > 256 MB

## 3. Użycie w kodzie (Rust)

Aby pobrać tymczasowy bufor z puli:

```rust
use crate::tensor::pool::TensorPool;

fn my_kernel(data: &[f16]) {
    // Pobierz bufor f32 o wymaganym rozmiarze
    let mut workspace = TensorPool::get_f32_buffer(data.len());
    
    // Wykonaj obliczenia...
    for i in 0..data.len() { workspace[i] = data[i].to_f32(); }
    
    // Bufor jest ZWRACANY do puli automatycznie, gdy 'workspace' wyjdzie poza zakres (Drop trait)
}
```

## 4. Zasady Bezpieczeństwa
1. **Zero-Copy**: `TensorPool` zwraca `&mut [T]` lub inteligentny wskaźnik, który gwarantuje, że dane nie są kopiowane podczas pobierania z puli.
2. **Alignment**: Wszystkie bufory są wyrównane do granic 64-bajtowych, co jest wymagane dla instrukcji AVX-512.
3. **Hot-Swap**: Jeśli wymagany rozmiar przekracza aktualną pojemność kubła, `TensorPool` dokona jednorazowej alokacji systemowej i zachowa ją w puli na przyszłość.

## 5. MSTS Integration
Podczas pracy w trybie **SSD Streaming (MSTS Path C)**, kafelki danych są ładowane bezpośrednio do buforów pochodzących z `TensorPool`. Dzięki temu proces strumieniowania 16GB modelu może odbyć się przy stałym śladzie pamięci RAM rzędu kilkuset megabajtów.
