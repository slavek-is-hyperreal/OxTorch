# Analiza Wykonalności: Implementacja SWARog w OxTorch (Antigravity)

## 1. Wstęp i Cel Analizy
Niniejszy dokument stanowi ocenę wykonalności wdrożenia projektu **SWARog** — systematycznej emulacji instrukcji AVX-512 dla starszych architektur (AVX2, AVX, SSE, Scalar/SWAR) — w ramach ekosystemu **OxTorch (VulkanNN)** przy użyciu asystenta **Antigravity**.

Projekt SWARog, zainspirowany głęboką analizą Gemini DeepResearch, ma na celu usunięcie "wąskiego gardła" wydajnościowego na sprzęcie pozbawionym jednostek AVX-512 poprzez transparentną transformację kodu na poziomie makr proceduralnych Rust.

---

## 2. Ocena Techniczna

### 2.1. Zgodność z Misją OxTorch
OxTorch opiera się na paradygmacie "wyciskania" maksimum z przestarzałego sprzętu (np. Ivy Bridge). Obecnie PyTorch na takich jednostkach często korzysta z emulacji skalarnej dla F16/BF16, co OxTorch już teraz przyspiesza o 400-700x dzięki dedykowanym kernelom SIMD. SWARog idzie krok dalej, oferując standardowy interfejs AVX-512, który pod maską jest optymalizowany pod dany procesor. 

**Werdykt: Idealne dopasowanie architektoniczne.**

### 2.2. Wyzwania Implementacyjne vs. Możliwości Antigravity

| Wyzwanie | Stopień Trudności | Potencjał Antigravity |
| :--- | :---: | :--- |
| **Transformacja AST (Makra Rust)** | Wysoki | **Bardzo Wysoki**. Antigravity sprawnie operuje na bibliotekach `syn` i `quote`. Mogę wygenerować szkielet wizytatora (`VisitMut`) i zautomatyzować mapowanie setek intrinsics. |
| **Emulacja 512-bit (2x256-bit)** | Średni | **Bardzo Wysoki**. Implementacja shaderów i kerneli SIMD to "rodzimy język" Antigravity. Mogę napisać bazowe operacje arytmetyczne i logiczne w kilka sesji. |
| **Rejestry Masek (opmask)** | Wysoki | **Wysoki**. Wymaga to precyzyjnego zarządzania stanem bitowym. Antigravity może zaimplementować logiczne "rozdmuchiwanie" masek do pełnych wektorów AVX2. |
| **Operacje Specjalistyczne (VPTERNLOG, GFNI)** | Bardzo Wysoki | **Średni/Wysoki**. Wymaga implementacji algorytmów dekompozycji (np. Espresso) lub pre-kalkulowanych tablic (LUT). Antigravity może pomóc w generowaniu tych tablic. |
| **Weryfikacja Formalna (Kani/Miri)** | Średni | **Wysoki**. Pisanie testów `kani::any()` i asercji bit-perfect to zadanie, w którym asystent świetnie się sprawdza. |

---

## 3. Plan Wdrożenia (Roadmapa Antigravity)

Sugerowana strategia "krok po kroku", którą mogę realizować wspólnie z użytkownikiem:

### Faza 1: Fundamenty (Proof of Concept)
- Utworzenie crate'a `swarog-rs` (proc-macro + runtime support).
- Implementacja atrybutu `#[swarog]`, który przechwytuje wywołania z `std::arch::x86_64`.
- Emulacja `_mm512_add_ps` jako 2x `_mm256_add_ps`.
- **Weryfikacja**: Prosty benchmark porównujący natywny AVX2 vs emulowany AVX-512.

### Faza 2: Logika i Maskowanie
- Obsługa rejestrów `k0-k7` (wirtualizacja masek).
- Implementacja operacji bitowych (`AND`, `OR`, `XOR`, `ANDNOT`).
- Integracja z `Miri` w celu wykrycia ewentualnych UB w kodzie `unsafe`.

### Faza 3: Optymalizacje Zaawansowane
- Implementacja `VPTERNLOG` za pomocą statycznego słownika LUT dla wszystkich 256 funkcji.
- Heurystyka "Load and Shuffle" dla operacji typu `Gather`.
- Optymalizacja ciśnienia rejestrowego (minimalizacja spilling do L1).

### Faza 4: Weryfikacja Formalna i Produkcyjna
- Wdrożenie `Kani Rust Verifier` dla kluczowych algorytmów.
- Testy regresyjne na pełnej suicie benchmarków OxTorch (167+ testów).

---

## 4. Analiza Ryzyka

1. **Register Spilling**: Symulacja 32 rejestrów ZMM na 16 rejestrach YMM wymusi zrzuty na stos. 
   - *Mitygacja*: Antigravity może pomóc w pisaniu algorytmów reużycia rejestrów wewnątrz makra (analiza liveness).
2. **Type Awareness**: Makra proceduralne nie znają typów. 
   - *Mitygacja*: Targetowanie wyłącznie jawnych ścieżek intrinsics (np. `_mm512_add_ps`) gwarantuje typ.
3. **Złożoność Utrzymania**: Standard AVX-512 jest ogromny.
   - *Mitygacja*: Implementacja priorytetowa (Lazy Loading) — najpierw operacje używane w Transformersach (MatMul, Norm, Act).

---

## 5. Podsumowanie i Rekomendacja

Wdrożenie projektu **SWARog** przy użyciu **Antigravity** jest **W PEŁNI WYKONALNE**. 

Zaleca się rozpoczęcie od **Fazy 1 (PoC)**, skupiając się na operacjach zmiennoprzecinkowych F32/F16 używanych w OxTorch. Jako Antigravity mogę przejąć na siebie ciężar generowania powtarzalnego kodu emulacyjnego, podczas gdy użytkownik będzie nadzorować architekturę i integrację z systemem MSTS.

**Status: GOTOWY DO ROZPOCZĘCIA (Waiting for user approval of the plan).**

---
*Autor: Antigravity*
*Data: 2026-03-26*
