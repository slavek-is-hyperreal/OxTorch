# Plan Pracy Badawczej: Kaskadowa Faktoryzacja BitNet (SBC)
**Projekt: Sequential BitNet Cascade (SBC) Distillation**
**Cel: Komercyjna optymalizacja LLM dla systemów legacy (OxTorch/MSTS)**

---

## 1. Wstęp i Cel Badania
Celem projektu jest udowodnienie, że zastąpienie standardowej warstwy liniowej (FP16/BF16) przez **sekwencyjną kaskadę dwóch warstw ternarnych (1.58-bit)** wspieraną przez nisko-wymiarowy adapter (LoRA) pozwala na uzyskanie wyższej wierności modelu (niższy MSE) przy zachowaniu drastycznie mniejszego śladu pamięciowego i lepszej charakterystyki strumieniowania danych (MSTS).

Projekt ma charakter komercyjny – wyniki posłużą do budowy lekkich modeli Bielik-OxTorch oraz jako podstawa publikacji naukowej.

---

## 2. Architektura: Sequential BitNet Cascade (SBC)

Tradycyjna warstwa $Y = W \cdot X$ zostaje zastąpiona przez moduł:
$$Y_{SBC} = \text{BitNet}_2( \text{BitNet}_1(X) ) + \text{LoRA}(X)$$

### Komponenty:
1.  **BitNet_1 (The Foundation):** Warstwa ternarna $\{-1, 0, 1\}$. Inicjalizowana przez naiwną kwantyzację wag oryginalnego modelu Bielik.
2.  **BitNet_2 (The Contextualizer):** Druga warstwa ternarna, inicjalizowana losowo. Ma za zadanie przejąć nieliniowości i błędy kwantyzacji pierwszej warstwy.
3.  **LoRA (The Error Multi-Bridge):** Równoległy adapter o niskim rzędzie ($r=32$), działający w wysokiej precyzji (BF16), korygujący finalny wynik.

> [!TIP]
> **Zaletą SBC** jest "streamability". Dwa mniejsze kernele ternarne są łatwiejsze do kafelkowania (tiling) w cache L1/L2 niż jedna olbrzymia macierz FP16.

---

## 3. Protokół Eksperymentalny (Phase-wise Distillation)

Eksperyment zostanie przeprowadzony w środowisku chmurowym (RunPod/Vast.ai) na GPU klasy RTX 3090/4090.

### Krok 0: Snapshot Aktywacji (Data Collection)
- Uruchomienie modelu Bielik-11B w precyzji BF16.
- Zbiór danych: Polski podzbiór CulturaX (ok. 5000 próbek).
- Zrzut par $(\text{input}, \text{output})$ dla wybranych warstw (np. warstwy 5, 10, 15 - MLP UpProj).

### Krok 1: Faza "Warm-up" (Frozen Base)
- **Status:** `BitNet_1` jest **ZABLOKOWANY** (wagi ze skwantowanego oryginału).
- **Cel:** Trening `BitNet_2` i `LoRA`, aby nauczyły się korygować błędy bazy.
- **Optymalizator:** AdamW z wysokim LR dla LoRA i umiarkowanym dla BitNet2.

### Krok 2: Faza "Fine-tuning" (Full Unfreeze)
- **Status:** Wszystkie parametry (`BitNet_1`, `BitNet_2`, `LoRA`) są **ODBLOKOWANE**.
- **Cel:** Subtelne przesunięcie wag bazowych, aby dostosować je do nowo nauczonego korektora.
- **Strategia:** Differential Learning Rates (10x mniejszy LR dla `BitNet_1`).

---

## 4. Logistyka i Budżetowanie

Badanie zostanie przeprowadzone metodą **Rolling Window** (warstwa po warstwie), co minimalizuje koszty VRAM i dysku.

| Zadanie | Estymowany czas | Koszt (RTX 3090/4090) |
| :--- | :--- | :--- |
| **Proof of Concept (1 warstwa)** | 6 - 8 godzin | **~$2 - $4** |
| **Pełny model (48 warstw)** | ~350 godzin | **~$70 - $90** |
| **Przechowywanie danych (200GB)** | 1 miesiąc | **~$15** |

**RAZEM (Badanie Pełne): ok. 400 - 500 PLN.**

---

## 5. Metryki Sukcesu (Do Publikacji)

Aby praca badawcza miała wartość komercyjną i naukową, zmierzymy:
1.  **MSE Loss:** Jak blisko oryginału jest wynik SBC.
2.  **KLD (Kullback-Leibler):** Zachowanie dystrybucji prawdopodobieństwa (Logits).
3.  **Efficiency Ratio:** Teoretyczny zysk na przepustowości w architekturze MSTS przy użyciu kaskady.
4.  **Token Parity:** % zgodności tokenów top-1 z modelem bazowym.

---

> [!IMPORTANT]
> **Następne działanie:** Przygotowanie skryptu `distill_cascade.py` w PyTorch, który automatyzuje Fazę 1 i Fazę 2.
