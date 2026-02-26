# Changelog

Wszystkie znaczące zmiany w projekcie VNN będą dokumentowane w tym pliku.

## [Unreleased] - Faza 5 i 6: VulkanNN Rusted Ed
Wprowadzono natywną bibliotekę **VulkanNN Rusted Ed** napisaną w języku Rust, zaprojektowaną jako 1:1 "drop-in replacement" dla `vulkan_nn_lib`. 
Rozwiązanie to usuwa wąskie gardła interpretera Pythona podczas operacji out-of-core.

### Added (Rust)
- **Moduł `vulkannn_rusted`**: Całkowicie natywna biblioteka zbudowana za pomocą `PyO3` oraz systemu `maturin`.
- **Backend WGPU**: Napisane w czystym WGSL shadery obslugujace operatory Dodawania (Add), Mnożenia Macierzy (MatMul) oraz funkcji aktywacji (ReLU, Sigmoid, SiLU). Wielo-wymiarowe grupy robocze WGPU znoszące obostrzenie 64k alokacji dispatcha pozwalające na nielimitowane rozmiary tablic.
- **Ekstremalnie szybki potok DMA (Tiered Memory Cache)**: 
  - **L3 Cache (Dysk)**: Zastosowano zerowe kopiowanie przez `memmap2`.
  - Zaawansowany OS-level prefetching przy uzyciu POSIX `madvise(MADV_WILLNEED)`, rozkazujący jądru Linuksa załadowanie danych do RAM asynchronicznie przez kontroler dysku DMA.
  - **L1 Cache (VRAM)**: Użycie buforów zwalnianych mechanizmem "Ping-Pong" (buforowanie strumieni / recycling buforów WGPU), aby zniwelować podskoki czasu opóźnień sterownika podczas per-operacyjnej alokacji danych wejściowych i wyjściowych na karcie graficznej.
- **Testy wydajności**: Skrypty `bench_table.py` i `benchmark_rust_vs_py.py` weryfikujące poprawność arytmetyczną wg NumPy (100% dokładność Parity ✅) oraz wydajność operacji.
- **Pełne Parity API z Pythonem**: Obsługa operacji logicznych (+, @) i klasa `Tensor(data)` na wzór odpowiednika Pythonowskiego.

### Changed (Python `vulkan_nn_lib` i reszta)
- Naprawiono precyzję numeryczną w Pythonowym backendzie Taichi. Zmieniono `k_reduce_sum` uniemożliwiające błędne zwracanie 0.0 w redukcjach z typem `float32`.
- Zintegrowano operacje dla **Trybu Kaggle**. Znacznie rozbudowano możliwości delegowania obliczeń `MatMul` offline po przekroczeniu progu GB do wirtualnej super-maszyny przez API Kaggle, z jednoczesnym wsparciem wznawiania sesji.
- Zoptymalizowano `to_numpy()` do szybkiego dostępu (fast path bypass).
- Zaktualizowano tryb CPU pozwalający na utrzymywanie potężnych tensorów w samej pamięci RAM redukując zuzycie pamięci. 

### Fixed
- Naprawiono błędy `AttributeError` dla tensora powiązanego z pamięcią RAM (RAM-resident array).
- Rozwiązano błędy spadku VRAM do poniżej 2GB (usunięcie limitera 512MB RAM dla silnika w systemach starszych GPU), optymalizując wyciąganie maksimum możliwości zaawansowaniem detekcji budżetu. 
