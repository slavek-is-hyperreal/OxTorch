# Notatki / "Zaparkowane" Pomysły (ADHD Parking Lot)

Ten plik służy jako brudnopis i przechowalnia pomysłów architektonicznych, które wpadają do głowy w trakcie trwania sprintów. Będziemy do nich cyklicznie wracać i decydować, czy wdrażamy je do głównej `roadmap.md`, czy wyrzucamy.

---

## 1. MSTS: Wielki "Kondensator" FIFO w RAM-ie dla dysków SSD
**Gdzie:** Sprint 4 / 5 (Offloading inferencji dużych modeli LLM)
**Kiedy dodano:** Koniec Fazy 1 (Indexing)

### Problem:
Obecny bufor pierścieniowy (Ring Buffer) MSTS offloaduje wagi pobierane z SSD prosto do małych obszarów optymalizowanych pod Cache procesora (L2/L3). Jeżeli dysk zaliczy tzw. "I/O spike" (nagły spadek prędkości / opóźnienie w stronicowaniu), wątek obliczeniowy zostaje "zagłodzony" i czeka na dane. Równocześnie w komputerach z kilkoma GB wolnego fizycznego RAMu (np. na maszynach z systemem na zapleczu) ta pamięć nie jest w żaden sposób używana przy offloadingu.

### Rozwiązanie:
- Zbudować asynchroniczną, gigantyczną kolejkę FIFO (Look-ahead Prefetching Buffer) alokowaną w wolnym fizycznym RAM-ie (np. zajmującą 4GB z wolnych 5.6GB).
- Kolejka działa jak **kondensator** wygładzający nierówności w transferze z SSD.
- Dedykowany wątek roboczy (np. z wykorzystaniem `io_uring`) czyta dane mechanicznie i pożera wagi z dysku z maksymalną przepustowością, próbując zawsze trzymać RAM-owy kondensator wypełniony (np. wagami LLaMY dla następnych wywołań sieci).
- Wątki obliczeniowe procesora / CPU Workery (i ich Cache) pobierają dane bez żadnego czekania (zero latency offloading) prosto z RAM-u, a nie bezpośrednio z mmapowanego dysku.

### Korzyści:
Gwarantowane wygładzenie dropów wydajnościowych dysków SSD/NVMe. Pozwala uzyskać stałą liczbę generowanych *tokenów na sekundę* bez nagłych "zaciąć" inferencji LLM wynikających z opóźnień systemu plików operacyjnego.

---

## 2. Vulkan Descriptor Pool Memory Expansion
**Kiedy dodano:** Koniec Fazy 1 (Indexing)

### Wniosek:
Rozszerzanie backendu o kolejne potoki (pipelines) takie jak `index_select` szybko wysusza `vk::DescriptorPoolCreateInfo`. Alokujemy pule z góry (`create_pool()`). Należy pamiętać o ciągłym monitorowaniu `max_sets` (aktualnie podbite do 1024) oraz `descriptor_count` (podbite do 4096 dla STORAGE), w przeciwnym razie nowo dodane operacje zwrócą `ERROR_OUT_OF_POOL_MEMORY` na produkcji.
