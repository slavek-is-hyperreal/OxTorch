# OxTorch — API Reference (v3.8.1-rc)

This document covers both the native `vulkannn_rusted` API and the `oxtorch` drop-in package.

---

## 1. The `oxtorch` Drop-in Package

The `oxtorch` Python package (located in `vulkannn_rusted/oxtorch/`) provides a transparent PyTorch-compatible interface.

```python
import oxtorch as torch   # single import change

x = torch.randn(2048, 2048, dtype=torch.bfloat16)  # OxTorch Native
y = torch.linear(x, weight, bias)                 # 26x faster than PT
```

### Fallback Mechanism: The Proxy
OxTorch uses a **Dynamic Attribute Proxy** (`__getattr__`):
1.  **Native Check**: Calls the optimized Rust kernel if implemented.
2.  **SSD Guard**: If the tensor is on SSD, it refuses to pull to RAM for fallback (preventing OOM) unless `.to_ram()` is called.
3.  **RAM Fallback**: Converts to NumPy -> PyTorch -> Result.

---

## 2. DataType Enum

| Variant | Python Alias | Description |
|:---|:---|:---|
| `DataType.F32` | `torch.float32` | 32-bit IEEE 754 float |
| `DataType.F16` | `torch.float16` | 16-bit half-precision |
| `DataType.BF16` | `torch.bfloat16` | Brain Float 16 (LLM standard) |
| `DataType.Int8` | `torch.int8` | 8-bit signed integer |
| `DataType.BitNet2` | `bitnet2` | 2-bit quantization |
| `DataType.BitNet1_6` | `bitnet1.6` | 1.58-bit ternary quantization |

---

## 3. Tensor Class (Native `vulkannn_rusted.Tensor`)

### Factories
- **`zeros(shape, dtype, device)`**: Allocates zero-filled memory.
- **`ones(shape, dtype, device)`**: Allocates one-filled memory.
- **`rand(shape, dtype, device)`**: Allocates random memory.
- **`from_ssd(path, shape, dtype)`**: Maps a raw binary file via `io_uring`.
- **`new_ssd(path, shape, dtype)`**: Creates a new mapped file for output.
- **`to_vulkan()`**: Moves the tensor to GPU VRAM (using the high-speed `GpuOnly` pool).
- **`to_cpu()`**: Moves the tensor back to Host RAM (reclaims VRAM immediately).

### Operations
- **`linear(weight, bias=None)`**: High-performance linear layer (SIMD).
- **`matmul(other)`** / **`bmm(other)`**: Matrix multiplication / Batch MatMul.
- **`layer_norm(normalized_shape, weight, bias, eps)`**: Optimized normalization.
- **`rms_norm(normalized_shape, weight, eps)`**: Transformer-optimized norm.
- **`sum(dim=None)`** / **`mean(dim=None)`**: Reductions with **f64 accumulation**.
- **`relu()`**, **`gelu()`**, **`silu()`**, **`sigmoid()`**: SIMD activations.

### SSD Streaming Special Methods
- **`prefetch()`**: Spawns a background thread to fill the 512MB RAM Capacitor.
- **`msts_pytorch_apply(callback)`**: Streams a PyTorch function tile-by-tile over an SSD tensor, preventing OOM.

---

## 4. Hardware Memory Model

- **Capacitor**: Global RAM reservoir (50% RAM) for prefetching.
- **Crook Ring**: Local 8MB triple-buffered recycler (Zero-Allocation).
- **Alignment**: Hard-enforced **1MB** alignment for optimal ZFS/Direct I/O throughput.

---
*Reference version: v8.2 "Iron Age" (2026-04-01)*
