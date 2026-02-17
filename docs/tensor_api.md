# Tensor API: Deep Dive

The `Tensor` class is the heart of VNN. It is designed to behave like a `torch.Tensor` but manages a 3-way bridge between **Vulkan**, **CPU**, and **SSD**.

## Initialization

### `__init__(data=None, shape=None, device='auto', dtype=None, external_path=None)`
- **`data`**: Initial data (NumPy array, list, or scalar).
- **`device`**: 
    - `'vulkan'`: GPU acceleration (Taichi).
    - `'cpu'`: RAM-based (NumPy).
    - `'ssd'`: Disk-native (Memmoped via `TensorStore`).
    - `'auto'`: (Default) Automatically selects based on size and available RAM.
- **`external_path`**: If provided, the tensor mounts an existing binary file on disk (Zero-copy).

---

## Core Operations

### `to_numpy()`
Forces the tensor into a NumPy array in RAM. 
> [!WARNING]
> Calling this on a "Monster Scale" (e.g. 100GB) tensor will cause an OOM crash. Use slicing or `permute` instead.

### `backward(grad=None)`
Triggers the Autograd engine. It builds a topological sort of the graph and computes gradients.
- **SSD Support**: Fully mature. Handles multi-gigabyte gradient accumulation on SSD-resident parameters using the ARAS engine.

### `zero_grad()`
Clears the gradient buffer. 
- **Device-Aware**: If the tensor is on SSD, the gradient is initialized as an SSD tensor to save RAM.

### `item()`
Returns the Python scalar value of a 0D or 1-element tensor. 
- **Parity**: Matches PyTorch's `item()`.

---

## Identity & Graphing

### Hashability
Tensors are hashable by their **identity** (`id(self)`). This allows them to be used in sets and dictionaries during topological sorting of the computation graph, even if their content changes.

---

## Shape Manipulation (SSD-Optimized)

### `permute(*dims)`
Changes the dimension order. 
- **SSD Behavior**: Performs a **physical re-layout** on disk if the tensor is SSD-resident. This is different from PyTorch (which uses strides) but ensures that subsequent streaming operations are sequentially optimal.

### `expand(*shape)`
Mimics PyTorch's `expand`. 
- **Auto-Broadcasting**: VNN gradients automatically handle broadcasting during `backward()` to match expanded shapes.

---

## Boolean Masking & Conditional Logic

### `masked_fill(mask, value)`
- **Description**: Fills elements of `self` where `mask` is True with `value`.
- **SSD Strategy**: Uses **RAM-First Caching**. If the mask fits in RAM, it's cached once to avoid SSD thrashing. The main data remains streamed from disk.

### Comparison Operators (`>`, `<`, `==`, etc.)
All standard operators are implemented. 
- **Result**: Returns a new tensor where elements are `1.0` (True) or `0.0` (False). 
- **Broadcasting**: Supports scalar-tensor broadcasting (e.g., `tensor > 0.5`) across all backends.

---

## Math Backend Selection Logic

| Operation | Vulkan (<128MB) | CPU (Large) | SSD (Monster) |
| :--- | :--- | :--- | :--- |
| **Add/Sub/Mul** | Taichi Kernel | NumPy | ARAS Tiled Stream |
| **Matmul** | Taichi Kernel | NumPy | Block-Based Parallel Stream |
| **Exp/Log/Pow** | Taichi Kernel | NumPy | SOE Ufunc Stream |
| **Sum/Mean** | - | NumPy | ARAS Tiled Reduction |
