# Rust Error Handling Workshop: Agent's Guide

To zaawansowany przewodnik dla agentów AI pracujących nad projektami w języku Rust. Celem tego cyklu pracy jest osiągnięcie czystej kompilacji (zero błędów, zero ostrzeżeń) w sposób systematyczny i udokumentowany.

---

## 🛠 Cykl Pracy (Workflow)

Każda zmiana w kodzie Rust powinna przejść przez poniższy cykl:

1.  **Kompilacja**: Uruchom `cargo check` (lub `cargo build`).
2.  **rustc --explain**: Dla każdego błędu (np. `E0609`) uruchom narzędzie objaśniające:
    ```bash
    rustc --explain E0609
    ```
4.  **Wyjaśnianie Warningów**: Większość ostrzeżeń to tzw. *linty* (np. `dead_code`). Choć nie mają one kodów `E`, kompilator podaje sekcję `help:` z rozwiązaniem. W przypadku skomplikowanych warningów użyj:
    ```bash
    cargo clippy -- -W clippy::all
    ```
5.  **Dokumentacja Analizy**: Przed przystąpieniem do naprawy, udokumentuj błąd/ostrzeżenie i plan naprawy w pliku:
    `/tmp/rust_error_analysis.md`
6.  **Naprawa**: Wprowadź poprawki w kodzie na podstawie planu.
7.  **Iteracja**: Powtarzaj kroki 1-6, aż kompilacja przejdzie **bez błędów i ostrzeżeń**.
8.  **Czyszczenie**: Po sukcesie wyczyść treść pliku analizy, zostawiając jedynie informację o jego przeznaczeniu.

---

## 🔍 Narzędzia Wspomagające (MCP)

### 1. `rust-docs` (docs.rs) 📖
Kiedy `rustc --explain` to za mało, użyj `rust-docs`, aby sprawdzić specyfikację typów i metod konkretnych bibliotek.
*   **Wyszukiwanie**: `mcp_rust-docs_search_documentation_items(crate_name="ash", keyword="PhysicalDeviceFloat16Int8Features")`
*   **Eksploracja**: Przejrzyj stronę indeksową crate'a, aby zrozumieć hierarchię modułów.

### 2. `github-mcp-server` (Search) 🐙
Gdy mierzysz się z rzadkim błędem lub specyficznym użyciem API (np. niskopoziomowy Vulkan/Ash), sprawdź, jak poradzili sobie z tym inni w "fajnych" projektach (np. `wgpu`, `bevy`, `burn`).
*   **Wyszukiwanie kodu**: `mcp_github-mcp-server_search_code(q="shader_float16 ash vk language:rust")`
*   **Wnioskowanie**: Podatność na błędy w Ash często wynika ze zmian wersji API. Szukaj najnowszych przykładów.

---

## 📝 Format `/tmp/rust_error_analysis.md`

Podczas analizy trzymaj się następującego formatu:

```markdown
# Analiza Błędów Rust - [Data/Godzina]

## Błędy (Errors)
- **ID**: `E0609`
- **Plik**: `src/sys_info.rs:106`
- **Opis**: Brak pola `shader_float16` w tej wersji `PhysicalDeviceFeatures`.
- **Analiza**: W Ash v0.38+ flagi FP16 są w rozszerzeniu `PhysicalDeviceFloat16Int8Features`.
- **Plan Naprawy**: Użyć `instance.get_physical_device_features2` z dołączoną strukturą rozszerzeń.

## Ostrzeżenia (Warnings)
- [ ] Unused variable `n` w `linalg.rs:273` -> Prefiks `_n`.
```

---

> [!TIP]
> **Zero Warnings Policy**: Projekt OxTorch dąży do braku ostrzeżeń. Warning to błąd, który jeszcze nie wybuchł. Traktuj je z taką samą powagą jak błędy kompilacji.
