# Tensor API: Deep Dive

The `Tensor` class is the heart of VNN. It is designed to behave like a `torch.Tensor` but manages a 3-way bridge between **Vulkan**, **CPU**, and **SSD**.

## Initialization

### `__init__(data=None, shape=None, requires_grad=False, device='auto', dtype=None, external_path=None)`
- **`data`**: Initial data (NumPy array, list, or scalar).
- **`device`**: 
    - `'vulkan'`: Taichi acceleration.
    - `'cpu'`: PyTorch Hybrid Fast-Path (uses zero-copy shared memory).
    - `'ssd'`: Disk-native storage via `TensorStore`.
    - `'kaggle'`: Cloud-native remote execution.
    - `'auto'`: (Default) Dynamically switches based on size and global RAM budget.
- **`external_path`**: If provided, the tensor mounts an existing binary file on disk (Zero-copy).
- **`requires_grad`**: Enables Autograd tracking.

---

## Core Operations

### `to_numpy()`
Forces the tensor into a NumPy array in RAM. 
> [!WARNING]
> Calling this on a "Monster Scale" (e.g. 100GB) tensor will cause an OOM crash. Use slicing or `permute` instead.

### `backward(grad=None)`
Triggers the Autograd engine.
- **SSD Support**: Fully mature with **Adaptive Restart**. If a backward pass causes a memory spike, DRAS v4 will automatically throttle prefetching or restart with a smaller tile size to guarantee stability.

### `zero_grad()`
Clears the gradient buffer. 
- **Device-Aware**: If the tensor is on SSD, the gradient is initialized as an SSD tensor to save RAM.

### `item()`
Returns the Python scalar value of a 0D or 1-element tensor. 

### `relu()`, `silu()`, `leaky_relu(alpha)`, `gelu_tanh()`, `softmax(dim)`
Standard activation functions implemented as direct methods on the `Tensor` object.
- **SOE Support**: These operations are fully streamed when the tensor resides on SSD.
- **Parity**: Matches PyTorch behavior exactly (100% verified parity).

### `pow(other)` or `**`
Element-wise exponentiation. Supports both scalar and tensor exponents.
- **Type Promotion**: Automatically promotes results to `float32`.

---

## Identity & Graphing

### Hashability
Tensors are hashable based on their **identity** (`id(self)`). This enables their use in sets and dictionaries during topological sorting of the computation graph, even if their content is modified.

---

## Shape Manipulation (SSD-Optimized)

### `permute(*dims)`
Changes the dimension order. 
- **SSD Behavior**: Executes a **physical re-layout** on disk if the tensor is SSD-resident. Unlike PyTorch (which manipulates strides), this ensures that subsequent streaming operations remain sequentially optimal.

### `expand(*shape)`
Mimics PyTorch's `expand` functionality. 
- **Auto-Broadcasting**: VNN's Autograd engine automatically handles broadcasting during `backward()` to match expanded shapes.

### `reshape(*shape)` / `view(*shape)`
Returns a new tensor with the same data but a different shape. 
- **Efficiency**: Zero-copy if possible. For SSD tensors, it returns a virtual view.

### `flatten(start_dim=0, end_dim=-1)`
Flattens the tensor into a 1D or lower-dimensional representation. 

---

## Boolean Masking & Conditional Logic

### `masked_fill(mask, value)`
- **Description**: Fills elements of `self` with `value` where `mask` is True.
- **SSD Strategy**: Employs **RAM-First Caching**. If the mask fits in RAM, it is cached to avoid SSD thrashing, while the primary data remains streamed from disk.

### Comparison Operators (`>`, `<`, `==`, etc.)
Standard comparison operators are fully supported. 
- **Result**: Returns a new tensor where elements are `1.0` (True) or `0.0` (False). 
- **Broadcasting**: Supports both scalar-tensor and tensor-tensor broadcasting across all backends.

---

## 🛠️ Specialized Access

### `__getitem__(idx)` / `__setitem__(idx, value)`
Standard Python indexing and slicing.
- **SSD Safety**: Slicing an SSD tensor returns a new virtual view or a small RAM-resident copy depending on context.
- **VRAM Caution**: Slicing large Vulkan tensors may trigger a sync.

### `get_samples(indices)`
High-performance retrieval of specific indices. Designed for OOM-safe parity verification and sparse data inspection.

---

## Math Backend Selection Logic

| Operation | Vulkan (<128MB) | CPU (Large) | SSD (Monster) | Kaggle (>1GB) |
| :--- | :--- | :--- | :--- | :--- |
| **Add/Sub/Mul** | Taichi Kernel | PyTorch (Shared) | SOE Tiled Stream | Remote GPU |
| **Matmul** | Taichi Kernel | PyTorch (Shared) | Block-Based Stream| Remote GPU |
| **Exp/Log/Pow** | Taichi Kernel | PyTorch (Shared) | SOE Ufunc Stream | Remote GPU |
| **Sum/Mean** | Taichi Kernel | PyTorch (Shared) | ARAS Tiled Reduc. | Remote GPU |
| **Activations** | Taichi Kernel | PyTorch (Shared) | SOE Streaming | Remote GPU |
