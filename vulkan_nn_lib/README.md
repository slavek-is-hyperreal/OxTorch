# Tensor Forever Library (`vulkan_nn_lib`)

A lightweight, high-performance neural network abstraction layer built on **Taichi Vulkan**.

---

## 🚀 Core Engine (`core.py`)

### 🛠 GPU Kernels (Vulkan)
Specialized kernels for parallel execution. Most kernels use 1D flattened indexing for zero-copy reshapes.

| Kernel | Description |
| :--- | :--- |
| `k_zero(X, Total)` | Zeros out a VRAM buffer. |
| `k_add(A, B, Total)` | $A = A + B$ (in-place). |
| `k_mul(A, B, Total)` | $A = A * B$ (in-place). |
| `k_matmul(A, B, C, M, N, K)` | Standard Matrix Multiplication. |
| `k_matmul_tiled(...)` | Paged Matrix Multiplication (Weight Paging). |
| `k_add_bias(X, bias, M, N)` | Element-wise bias addition. |
| `k_relu_1d(X, Total)` | Rectified Linear Unit. |
| `k_leaky_relu_1d(X, Total, alpha)` | Leaky ReLU. |
| `k_softmax_1d(X, M, N)` | Softmax over the last dimension. |
| `k_rmsnorm_1d(X, W, M, N, eps)` | Root Mean Square Layer Normalization. |
| `k_silu_1d(X, Total)` | Sigmoid Linear Unit (Swish). |
| `k_rope(X, cos, sin, B, L, H, D)` | Rotary Positional Embedding (LLMs). |
| `k_embedding_1d(...)` | Fast token-to-vector lookup. |
| `k_attention(...)` | Optimized Scaled Dot-Product Attention. |
| `k_kv_cache_update(...)` | Incremental KV-cache updates for chat. |
| `k_conv2d_1d(...)` | 2D Convolution on flattened buffers. |
| `k_pool2d_1d(...)` | Max Pooling 2D. |
| `k_upsample2d_1d(...)` | Nearest-neighbor upsampling. |
| `k_concat_1d(...)` | Tensor concatenation. |

---

### 📦 `Tensor` Class
The primary data structure. Wraps a `ti.ndarray` in VRAM.

- **Initialization**: `Tensor(data, shape=None, dtype=ti.f32)`
    - Can take `numpy` arrays, lists, or scalars.
    - If `data=None`, allocates an empty buffer of `shape`.
- **`to_numpy()`**: Downloads data from VRAM to a NumPy array.
- **`total_size`**: Property returning the flattened number of elements.
- **`reshape(*shape)` / `view(*shape)`**: Metadata-only reshape (zero-copy).
- **`squeeze(dim=None)`**: Removes dimensions of size 1.
- **`save_to_disk(path)`**: Saves raw binary to file.
- **`from_disk(path, shape, dtype)`**: Loads raw binary directly (via RAM).

---

### 🏛 `Module` Architecture
Base class for all layers, mimicking PyTorch.

- **`forward(*args)`**: Must be implemented by subclasses.
- **`state_dict()`**: Returns a dictionary of all parameters (NumPy).
- **`load_state_dict(sd)`**: Populates VRAM parameters from a dictionary.
- **`to(device)`**: Mock method for compatibility.

#### Available Layers (`vnn.nn`)
- **Linear**: `Linear(in, out, bias=True)`
- **TiledLinear**: `TiledLinear(in, out, tile_size=1024)` (Virtual VRAM).
- **MatFormerLinear**: `MatFormerLinear(...)` (Matryoshka slicing support).
- **RMSNorm**: `RMSNorm(dim, eps=1e-6)`
- **Softmax**: `Softmax(dim=-1)`
- **SiLU**: `SiLU()`
- **Embedding**: `Embedding(vocab, dim)`
- **Conv2d**: `Conv2d(in, out, kernel_size, bias=True)`
- **MaxPool2d**: `MaxPool2d(kernel_size, stride)`
- **Upsample**: `Upsample(scale_factor)`
- **ReLU / LeakyReLU**: Activation layers.
- **Sequential**: `Sequential(*layers)`
- **Concatenate**: `Concatenate()`

---

### 🌌 Gemma 3 (LLM) Components
Specialized modules for the Gemma 3 architecture.

- **`RoPE`**: Layer wrapping `k_rope`.
- **`Gemma3Block`**: A full Transformer block with MatFormer projections and KV-cache logic.
- **`Gemma3Model`**: The full 35-layer stack.
- **`get_cos_sin(seq_len, head_dim)`**: Helper to generate RoPE caches.

---

## ⚡ PyTorch Shim (`torch_shim.py`)
Allows existing PyTorch code to run on Vulkan with **zero code changes** (via hijacking).

### Usage
```python
import vulkan_nn_lib.torch_shim as torch
# 'torch' now points to VulkanNN!
```

### Supported Mappings
- **`torch.Tensor`**: Points to `vnn.Tensor`.
- **`torch.nn.Module`**: Points to `vnn.Module`.
- **`torch.nn.functional`**:
    - `relu`, `leaky_relu`, `max_pool2d`, `silu`, `softmax`.
- **`torch.nn`**:
    - All layers listed above (Linear, Conv2d, etc.).
    - `torch.nn.TiledLinear`: Custom Vulkan extension.
- **`torch.from_numpy(x)`**: Converts NumPy to Vulkan `Tensor`.
- **`torch.load(path)`**: Simplified loader (supports `.npy` metadata).
- **`torch.no_grad()`**: Context manager for inference.
- **`torch.device(name)`**: Accepts any string.

---

## 🛠 System Initialization
The library automatically selects the best backend:
1.  **Vulkan**: Primary target.
2.  **CPU**: Automatic fallback if Vulkan drivers are missing.

Initialize manually in `core.py`:
```python
ti.init(arch=ti.vulkan) # or ti.cpu / ti.cuda
```
