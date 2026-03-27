# Implementation Plan: CPU Backend Migration - Phase 0.5 Bridge & Phase 1.add

Ten plan skupia się na ukończeniu Phase 0 (infrastruktura) poprzez dodanie mostka (bridge) do `src/cpu_old`, aby projekt mógł się budować w trakcie migracji, oraz na kontynuacji migracji operacji `add`.

## User Review Required
> [!IMPORTANT]
> **Mechanizm Bridge**: Zastosujemy `pub use crate::cpu_old::*;` w `src/cpu/mod.rs`. Dzięki temu nowy backend `cpu` będzie serwować stare implementacje, dopóki nie zostaną one przesłonięte przez nowe, granularne wersje w `src/cpu/ops`.

## Proposed Changes

### [Phase 0] Infrastructure & Bridging

#### [MODIFY] [mod.rs](file:///my_data/gaussian_room/vulkannn_rusted/src/cpu/mod.rs)
- Dodanie re-eksportu z `cpu_old`, aby zapewnić kompatybilność wsteczną dla reszty kodu (`Tensor`, `models`, itp.).

#### [MODIFY] [mod.rs](file:///my_data/gaussian_room/vulkannn_rusted/src/cpu/ops/mod.rs)
- Re-eksportowanie nowych operacji z submodułów (np. `binary`).

#### [MODIFY] [mod.rs](file:///my_data/gaussian_room/vulkannn_rusted/src/cpu/ops/binary/mod.rs)
- Re-eksportowanie grup operacji (np. `add`).

---

### [Phase 1] Binary Operations - `add`

#### [MODIFY] [mod.rs](file:///my_data/gaussian_room/vulkannn_rusted/src/cpu/ops/binary/add/mod.rs)
- Implementacja dispatchera dla `add`, obsługującego różne precyzje i wybór kernela (Generic/AVX/NEON) za pomocą makr lub `cfg`.

#### [NEW] [bf16/mod.rs](file:///my_data/gaussian_room/vulkannn_rusted/src/cpu/ops/binary/add/bf16/mod.rs)
- Dispatcher dla precyzji BF16. Wybiera `add_bf16_avx` na x86 z AVX, `add_bf16_neon` na ARM, lub `add_bf16_generic` jako fallback.

#### [MODIFY] [add_bf16_avx.rs](file:///my_data/gaussian_room/vulkannn_rusted/src/cpu/ops/binary/add/bf16/add_bf16_avx.rs)
- Dokończenie implementacji AVX. Obecna wersja wymaga weryfikacji obsługi reszty (tail elements) lub zapewnienia, że dispatcher ją obsługuje.

## Verification Plan

### Automated Tests
- `cargo build` – musi przechodzić po dodaniu bridge.
- `cargo test` – weryfikacja poprawności implementacji `add_bf16`.

### Manual Verification
- Porównanie wyników nowej implementacji `add` z `cpu_old` przy użyciu skryptu parzystości.
