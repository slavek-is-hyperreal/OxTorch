# OxTorch Execution Modes

OxTorch obsługuje cztery główne strategie wykonawcze, które pozwalają silnikowi adaptować się do rozmiaru danych i dostępnej infrastruktury sprzętowej.

---

## 1. CPU Mode (`device="cpu"`)

Tryb zorientowany na niską latencję i maksymalne wykorzystanie instrukcji SIMD procesora.

*   **Izolacja SIMD-only**: Kernele w OxTorch są pisane jako funkcje jednowątkowe. Unikamy używania `rayon` czy `std::thread` wewnątrz pojedynczej operacji (np. `add_f32`), aby zapobiec masowej wymianie danych w cache L2/L3 (cache thrashing).
*   **Prefetching**: Wykorzystujemy `_mm_prefetch` (x86) lub `PRFM` (ARM), aby załadować kolejne linie cache przed ich faktycznym użyciem w pętli SIMD.
*   **Kiedy używać?**:
    - Małe i średnie tensory (< 4M elementów).
    - Operacje wymagające częstego dostępu losowego (np. skomplikowane redukcje).
    - Gdy GPU jest zajęte innymi zadaniami.

---

## 2. Vulkan GPU Mode (`device="vulkan"`)

Tryb wysokiej przepustowości wykorzystujący moc obliczeniową karty graficznej przez API Vulkan 1.2 (biblioteka `ash`).

*   **Raw Ash Setup**: Nie używamy wysokopoziomowych wrapperów. Bezpośrednio zarządzamy `CommandPool`, `PipelineLayout` i `DescriptorSet`.
*   **Timeline Semaphores**: Do synchronizacji asynchronicznej używamy semaforów osi czasu, co pozwala na śledzenie postępu GPU bez blokowania wątku głównego (zero-block polling).
*   **PCIe Bottleneck (Bonaire GCN 1.1)**: Na starszym sprzęcie (PCIe 3.0) koszty transferu do buforów `staging` wynoszą ok. 80ms.
*   **Break-even Point**: GPU staje się szybsze od CPU dopiero powyżej **4 194 304 elementów** (4M). Poniżej tego progu narzut na dyspacz i transfer przewyższa zysk z obliczeń.

---

## 3. Hybrid Mode (`device="hybrid"`)

Unikalny tryb OxTorch inspirowany architekturą MERA-400. Realizuje strategię "Race for Tiles".

*   **Mechanizm**: Operacja jest dzielona na kafelki (Tiles). Wątek GPU i wątki CPU (Rayon) korzystają ze wspólnego atomowego licznika `AtomicUsize`.
*   **Race for Tiles**: Każda jednostka sprzętowa (GPU, Core 0, Core 1...) "wyrywa" kolejny dostępny kafelek z kolejki, przetwarza go i prosi o następny.
*   **Zaleta**: System automatycznie balansuje obciążenie. Jeśli GPU jest wolne z powodu transferu PCIe, CPU przejmuje więcej kafelek. Jeśli CPU jest zajęte IO, GPU nadrabia zaległości. Sformatowane pod pełne nasycenie sprzętu (100% CPU i GPU load).

---

## 4. SSD Streaming Mode (`device="ssd"`)

Tryb "nieskończonej pamięci", pozwalający uruchamiać modele 70B+ na maszynach z 8GB RAM.

*   **io_uring + O_DIRECT**: Używamy najnowszego API Linuksa do asynchronicznego I/O. Dzięki flagi `O_DIRECT` omijamy systemowy Page Cache, co eliminuje podwójne buforowanie i nieprzewidywalne skoki zużycia RAM.
*   **Aligned 1MB Records**: Dane na SSD są wyrównane do 1MB, co idealnie mapuje się na `recordsize=1M` w systemie plików ZFS. Pozwala to na transfer DMA (Direct Memory Access) prosto z kontrolera dysku do buforów procesora.
*   **CrookScheduler**: Bufor kołowy (Ring Buffer) zarządza ruchem danych. Podczas gdy CPU liczy `Kafel N`, system w tle ładuje `Kafel N+1` z SSD i zapisuje `Kafel N-1`.
