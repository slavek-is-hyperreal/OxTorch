# OxTorch Python Package & Fallback Mechanism

Paczka `oxtorch` to wysokopoziomowy wrapper Pythona, który sprawia, że OxTorch jest w 100% kompatybilny z API PyTorcha jako drop-in replacement (`import oxtorch as torch`).

---

## 1. Architektura Wrapperów

System składa się z dwóch warstw:
1.  **`vulkannn_rusted`**: Binarny moduł skompilowany w Rust (PyO3). Zawiera niskopoziomową klasę `Tensor` i kernele SIMD/Vulkan.
2.  **`oxtorch`**: Czysty moduł Pythona (`vulkannn_rusted/oxtorch/`). Implementuje logikę dyspozycji, proxying oraz fallback do oryginalnego PyTorcha.

---

## 2. Dynamiczna Dyspozycja (Module-level)

W `oxtorch/__init__.py` znajduje się funkcja `__getattr__`, która przechwytuje wywołania globalne (np. `torch.relu(t)`):

1.  **Native Check**: Sprawdza, czy `vulkannn_rusted.Tensor` posiada daną metodę. Jeśli tak, wywołuje ją natywnie.
2.  **PyTorch Fallback**: Jeśli operacji brakuje w Rust, OxTorch:
    - Konwertuje argumenty `OxTorchTensor` na `torch.Tensor`.
    - Wywołuje oryginalną funkcję z zainstalowanego pakietu `torch`.
    - Pakuje wynik z powrotem w `OxTorchTensor`.

---

## 3. Proxy Tensor (`oxtorch/tensor.py`)

Klasa `Tensor` w `oxtorch` jest wrapperem przechowującym natywny obiekt w polu `self._vnn`. Kluczowym mechanizmem jest tutaj `__getattr__` na poziomie klasy:

```python
def __getattr__(self, name):
    # 1. Native?
    if hasattr(self._vnn, name):
        # ... wywołaj natywny kernel ...
    
    # 2. SSD Fallback?
    if self.device == "ssd":
        return self.msts_pytorch_apply(...)
    
    # 3. Standard PyTorch Fallback (PULLS TO RAM!)
```

---

## 4. Inteligentny Fallback SSD (`msts_pytorch_apply`)

To unikalna cecha OxTorch. Jeśli wykonujesz operację `erf()` na tensorze 100GB znajdującym się na SSD:
1.  OxTorch wie, że nie ma natywnego kernela `erf` dla SSD.
2.  Zamiast rzucać błędem lub wciągać 100GB do RAM (co spowodowałoby OOM), wywołuje `msts_pytorch_apply`.
3.  Dane są strumieniowane kafelkami (1MB) przez orkiestrację MSTS.
4.  Każdy kafel jest na chwilę rzutowany do PyTorcha, procesowany i zrzucany z powrotem na SSD.

**Efekt**: Pełna funkcjonalność PyTorcha na ogromnych danych przy minimalnym zużyciu RAM.

---

## 5. Konfiguracja Środowiska

Ponieważ `oxtorch` dostarczany jest jako folder wewnątrz repozytorium (lub wewnątrz wheel), a nie jako samodzielna paczka `pip`, należy poprawnie ustawić ścieżki:

```bash
# Wersja deweloperska (lokalna)
export PYTHONPATH=$PYTHONPATH:/sciezka/do/vulkannn_rusted
```

W kodzie Python wystarczy wtedy:
```python
import oxtorch as torch
x = torch.randn(10, 10, device="vulkan") # Natywny OxTorch
y = torch.erf(x)                        # Fallback do PyTorch
```
