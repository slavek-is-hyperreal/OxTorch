# MSTS: Mera Style Tiling System

MSTS to serce architektury OxTorch, inspirowane oscylatorową i asynchroniczną naturą minikomputera MERA-400. System ten pozwala na przetwarzanie tensorów o rozmiarach przekraczających dostępną pamięć RAM (Out-of-Core) poprzez strumieniowanie danych z dysku SSD bezpośrednio do jednostek obliczeniowych (CPU/GPU).

---

## 1. Native MSTS vs. MSTS PyTorch (Fallback)

> [!IMPORTANT]
> Zrozumienie różnicy między tymi dwoma mechanizmami jest kluczowe dla wydajności systemu.

| Cecha | Native MSTS (OxTorch) | MSTS PyTorch (Fallback) |
|:---|:---|:---|
| **Język** | Pure Rust (`src/tensor/msts.rs`) | Rust + Python (`oxtorch/tensor.py`) |
| **Kernele** | Autorskie SIMD (AVX/NEON) | Dowolna funkcja `torch.*` |
| **Wydajność** | Maksymalna (zero GIL, zero alokacji) | Niższa (GIL overhead, NumPy conversion) |
| **Przeznaczenie** | Produkcyjne operacje (ReLU, MatMul) | Szybkie wsparcie nowych opów / prototypy |
| **Obsługa dtypes** | Pełna (F32, F16, BF16, I8, Ternary) | Zależna od mapowania NumPy |

### Jak działa Fallback?
Mechanizm `msts_pytorch_apply` w Pythonie przechwytuje wywołania operacji, które nie mają jeszcze natywnego odpowiednika SSD. Rozcina tensor SSD na kafelki (zwykle 1MB), konwertuje kafel do NumPy, przesyła do PyTorcha, odbiera wynik i zapisuje z powrotem na SSD. Pozwala to na używanie `torch.erf()` czy `torch.exp()` na 100GB tensorach bez OOM.

---

## 2. Automat Stanów StatefulTile

Każdy "kafelek" w buforze kołowym (`CrookScheduler`) posiada atomowy znacznik stanu (`AtomicU32`), który steruje orkiestracją I/O:

1.  **`TILE_EMPTY` (0)**: Kafel jest pusty. Robot odczytu (PPU) może zacząć ładować dane z SSD.
2.  **`TILE_READING_FROM_DISK` (1)**: `io_uring` wypełnia bufor.
3.  **`TILE_READY_FOR_COMPUTE` (2)**: Dane są w RAM. Jednostka obliczeniowa (CPU/GPU) może przejąć kafel.
4.  **`TILE_COMPUTING` (3)**: Kernel wykonuje operacje na danych. Kolejne wątki nie mogą "ukraść" tego kafla.
5.  **`TILE_READY_FOR_WRITE` (4)**: Wynik obliczeń jest gotowy do zrzutu na dysk.
6.  **`TILE_WRITING_TO_DISK` (5)**: `io_uring` zapisuje dane do pliku `.ssd`. Po zakończeniu kafel wraca do stanu `EMPTY`.

---

## 3. 3-Path Dispatch (Automatyczny dobór ścieżki)

OxTorch nie używa jednej strategii dla wszystkich rozmiarów. Próg wydajnościowy jest "wypalany" w pliku binarnym podczas kompilacji (`build.rs`) na podstawie parametrów cache L2/L3 maszyny docelowej.

### Path A: Direct (Tensor < `DIRECT_MAX` ≈ 3MB)
*   **Mechanizm**: Bezpośredni odczyt `read_chunk` do `AlignedBuffer` i zapis.
*   **Zaleta**: Zero narzutu na tworzenie wątków, synchronizację atomową czy orkiestrację ring-bufora. Idealne dla małych wektorów parametrów.

### Path B: Single-thread (Tensor < 32MB)
*   **Mechanizm**: 1 wątek I/O (tylko odczyt), obliczenia wykonywane inline w głównym wątku.
*   **Ring Size**: `RING_SMALL` (2 kafelki).
*   **Tile Size**: `TILE_SMALL` (celowany w 75% L2 cache, aby dane były "gorące").

### Path C: Full CrookScheduler (Tensor >= 32MB)
*   **Mechanizm**: Pełna orkiestracja. 2 dedykowane wątki I/O (Background Read & Write) + Rayon parallel compute.
*   **Ring Size**: `RING_LARGE` (dynamicznie dobierany pod L3, max 8).
*   **Tile Size**: `TILE_LARGE` (zwykle 2-4MB, optymalne pod sequential burst SSD).

---

## 4. Orkiestracja CrookScheduler

Scheduler implementuje asynchroniczny model "Race for Tiles". W trybie `Path C`:
- Wątek **Reader** pędzi do przodu, zapełniając `EMPTY` kafelki z SSD.
- Wątek **Compute** (główny/Rayon) czeka na `READY_FOR_COMPUTE`, przetwarza i oddaje.
- Wątek **Writer** pędzi za Computem, zrzucając `READY_FOR_WRITE` na SSD.

Dzięki temu eliminujemy przestoje (I/O wait) — podczas gdy CPU liczy `Tile 5`, SSD ładuje `Tile 7` i zapisuje `Tile 3`.
