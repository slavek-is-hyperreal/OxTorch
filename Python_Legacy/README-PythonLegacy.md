# 🐍 VNN Python Legacy Edition (vulkan_nn_lib)

> [!CAUTION]
> **Status: LEGACY.**
> Ta wersja biblioteki (Python + Taichi) nie jest już aktywnie rozwijana. Cały główny rozwój został przeniesiony do natywnego silnika **VNN Rusted (Rust + WGPU)** dostępnego w głównym folderze repozytorium.

VNN Python Legacy to oryginalna, hobbystyczna implementacja biblioteki tensorowej, zaprojektowana do uruchamiania ogromnych modeli AI na starym sprzęcie z ograniczoną pamięcią RAM i VRAM.

## 📼 Filozofia Legacy
Wersja ta udowodniła, że możliwe jest trenowanie modeli 100GB+ na domowych PC poprzez traktowanie SSD jako pierwszorzędnej warstwy pamięci.

### Kluczowe Funkcje (Classic):
- **SSD-Native Autograd**: Backpropagation strumieniowane bezpośrednio z/na dysk.
- **DRAS v4 (Adaptive RAM-Aware Streaming)**: Realne monitorowanie RAM z adaptacyjnym backoffem.
- **Taichi Engine**: Obliczenia Vulkan napędzane przez bibliotekę Taichi (JIT-compiled SPIR-V).
- **Kaggle Mode**: Możliwość oddelegowania obliczeń MatMul > 1GB do chmury Kaggle.

## 🏗️ Architektura i API

Stara wersja dokumentacji technicznej znajduje się w folderze `docs-python/`:
- **[Architecture](docs-python/architecture.md)**: Szczegóły silnika ARAS i SOE.
- **[Tensor API](docs-python/tensor_api.md)**: Opis klas Tensor i dostępnych operacji.
- **[Technical Manual](docs-python/technical_manual.md)**: Walkthrough po kodzie źródłowym.

## 🚀 Przykład Użycia (Legacy)

```python
import Python_Legacy.vulkan_nn_lib.torch_shim as torch

# Tensor 10GB+ na SSD
w = torch.randn(1024, 1024, 2560, requires_grad=True) 

# Matematyka na dysku
loss = (w * 2.0).sum()
loss.backward()

print(f"Gradient na urządzeniu: {w.grad.device}") # -> 'ssd'
```

---
*Jeśli szukasz wydajności, przejdź do głównego folderu i użyj wersji Rust (`vulkannn_rusted`). Jest ona od 2.5x do 3.5x szybsza niż ta implementacja.*
