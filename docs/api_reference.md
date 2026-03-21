# OxTorch — API Reference (v3.7.0)

This document covers both the native `vulkannn_rusted` API and the `oxtorch` drop-in package.

---

## The `oxtorch` Drop-in Package

The `oxtorch` Python package (located in `vulkannn_rusted/oxtorch/`) provides a transparent
PyTorch-compatible interface. For most use cases, this is the recommended entry point.

```python
import oxtorch as torch   # single import change — that's it

x = torch.randn(1024, 1024)       # uses real torch.randn (fallback)
y = torch.relu(x)                 # routes to OxTorch CPU/Vulkan kernel
z = torch.matmul(x.half(), x.t()) # OxTorch F16 Vulkan — up to 25x faster
```

### Fallback Mechanism

`OxTorchTensor.__getattr__` intercepts every attribute access:
1. If the attribute exists natively on `vulkannn_rusted.Tensor` → call it, wrap result.
2. Otherwise → convert to `numpy` → call on real `torch.Tensor` → wrap result back.

Module-level `__getattr__` in `oxtorch/__init__.py` handles factory functions
(e.g., `torch.zeros`, `torch.randn`) the same way.

> **Requirement**: PyTorch must be installed. OxTorch accelerates — it does not replace when it can't.

### Running scripts with oxtorch

```bash
# PYTHONPATH must include the vulkannn_rusted directory (contains oxtorch/ inside it):
PYTHONPATH=/path/to/vulkannn_rusted python your_script.py
```

---

## DataType Enum

Source: `src/tensor/types.rs`

| Variant | Description |
|:---|:---|
| `DataType.F32` | 32-bit IEEE 754 float |
| `DataType.F16` | 16-bit IEEE 754 half-precision float |
| `DataType.BF16` | Brain Float 16 (8-bit exponent, 7-bit mantissa) |
| `DataType.Int8` | 8-bit signed integer |
| `DataType.Ternary` | 1.58-bit BitNet quantization (weights: -1, 0, 1) |

Python aliases in `oxtorch/__init__.py`:
```python
oxtorch.f32 == oxtorch.float32 == DataType.F32
oxtorch.f16 == oxtorch.float16 == DataType.F16
oxtorch.bf16 == oxtorch.bfloat16 == DataType.BF16
oxtorch.int8 == DataType.Int8
```

---

## Tensor Class (Native `vulkannn_rusted.Tensor`)

Source: `src/tensor/mod.rs`

### Constructor

```python
import vulkannn_rusted as vnn
t = vnn.Tensor(data=None, shape=None, dtype=DataType.F32, device="cpu", name="tensor")
```

- `data`: Optional `numpy.ndarray` (F32). Converted to target dtype at construction.
- `shape`: Alternative to data. Allocates a zero-filled tensor.
- `dtype`: Storage precision. Conversion uses runtime-dispatched SIMD.
- `device`: `"cpu"`, `"vulkan"`, `"hybrid"`, or `"ssd"`.
- `name`: Label used for SSD file paths and debug output.

### Static Methods

**`Tensor.from_ssd(path, shape, dtype=DataType.F32)`**

Maps an existing binary file as a read-only SSD tensor backed by `io_uring`/`O_DIRECT`.
The file must exist and be at least `prod(shape) * bytes_per_element` bytes.
Returns a tensor with `device="ssd"`.

**`Tensor.new_ssd(path, shape, dtype=DataType.F32)`**

Creates a new file on disk and maps it read-write. Used for out-of-core results
(e.g., the 16GB Monster ReLU benchmark). Aligned to 1MB ZFS recordsize boundaries.

**`Tensor.save_ssd(path) -> Tensor`** *(new in v3.7.1)*

Writes the tensor's raw bytes to `path` and returns a new SSD-mapped tensor backed by `io_uring`/`O_DIRECT`. Equivalent to calling `Tensor.new_ssd(path, ...)` and writing the data. Returns a tensor with `device="ssd"`.

---

### Matrix Operations

**`__matmul__(other: Tensor)`** — operator: `@` — Source: `src/tensor/linalg.rs`

Routes based on `device`:
- `"cpu"`: CPU SIMD kernels (matrixmultiply sgemm + F16C/NEON for F16/BF16).
- `"vulkan"`: SPIR-V compute shader (tiled 16×16 matmul kernel, F32 compute).
- `"hybrid"`: MSTS tile-pulling between CPU and Vulkan.
- `"ssd"`: Streams tiles from disk via MSTS + io_uring.

**`bit_linear(activation, weight, scale)`** — static method — `src/tensor/linalg.rs`

Specialized **BitNet 1.58-bit** linear layer.
- `activation`: Int8 tensor
- `weight`: Ternary tensor
- `scale`: F32 scale vector

Uses zero-multiplication accumulation for ternary weights (-1, 0, +1).

---

### Activation Functions

All activation functions dispatch through the same hybrid path:

| Method | Notes |
|:---|:---|
| `relu()` / `relu_into(out)` | AVX1 `vmaxps` (F32), F16C (F16), Rayon scalar (BF16/INT8) |
| `gelu()` | Tanh approximation; INT8 uses precomputed LUT |
| `sigmoid()` | Per-element `1/(1+exp(-x))` |
| `silu()` | `x * sigmoid(x)`, fused |
| `softmax(dim)` | Dequantized F32 compute for INT8; direct for F16/BF16/F32 |
| `tanh()` / `clamp(min, max)` | Dispatched via Vulkan activation shader |
| `leaky_relu(neg_slope)` | Native kernel |
| `elu(alpha)` | Native kernel |

For `device="hybrid"` tensors ≥ 4M elements: GPU thread + CPU SIMD thread race for tiles.

---

### Element-wise & Scalar Operations

```python
a + b         # element-wise add
a - b         # element-wise sub
a * b         # element-wise mul
a / b         # element-wise div
a + 2.0       # scalar broadcast
a * 3.0       # scalar broadcast
```

Source: `src/tensor/ops.rs`, `src/cpu/ops/binary/`

---

### Reduction Operations

```python
t.sum()                # full tensor sum (INT8: i64-exact; others: F32 accumulation)
t.mean()               # mean
t.max()                # scalar maximum
t.min()                # scalar minimum
t.softmax(dim=-1)      # row-wise softmax
```

Source: `src/tensor/reductions.rs`

---

### Other Methods

**`to_numpy()`** — Convert F16/BF16/Int8/F32 storage to F32 numpy array via Rayon parallel SIMD upcast.

**`transpose()`** — Returns a transposed view (no data copy, flips `is_transposed` flag).

**`shape`**, **`device`**, **`dtype`** — Read-only properties.

---

## Backend Module

Source: `src/backend.rs`

`init_backend()` creates the global `BACKEND: OnceLock<AshBackend>` singleton. Called once automatically during `import vulkannn_rusted`.

### Key Public Functions

| Function | Description |
|:---|:---|
| `execute_activation(input, op, dtype, is_hybrid)` | Full-tensor GPU activation |
| `execute_activation_chunked(input, output, offset, count, op, dtype)` | Tile-range GPU activation for MSTS |
| `execute_matmul(a, b, m, k, n, dtype, is_hybrid)` | Full matrix multiply on GPU |
| `execute_matmul_with_bias(a, b, bias, m, k, n, act_type, dtype)` | MatMul + optional bias + activation |
| `execute_add_into(a, b, out, dtype, is_hybrid, staging)` | Element-wise addition |
| `execute_elementwise(a, b, out, op, dtype)` | Element-wise mul/sub/div on GPU |
| `poll_async_ops()` | Recycles completed GPU operations |
| `poll_async_ops_until(wait_id)` | Blocks until timeline semaphore value is reached |

### Buffer Pool

`get_buffer(size, usage, label, cpu_visible)` — returns a `CachedBuffer` from the recycle pool or allocates from the 1GB VRAM pool. CPU-visible (staging) buffers use `CpuToGpu` memory; device buffers use `GpuOnly`.

`recycle_buffer(buf)` — returns buffer to cache for reuse.

---

## MSTS Scheduler

Source: `src/crook_scheduler.rs`, `src/tensor/msts.rs`

`StatefulTile` — a 1MB-aligned structure with `AtomicU32` status field representing:

```
TILE_EMPTY → TILE_LOADING → TILE_READY_FOR_COMPUTE → TILE_COMPUTING → TILE_READY_FOR_WRITE → TILE_EMPTY
```

Workers poll and claim tiles via Compare-And-Swap without any mutex.

### 3-Path Dispatch (v3.7.1+)

`unary_op_ssd` and `save_ssd` automatically select the optimal path based on tensor size:

| Path | Size Threshold | IO Workers | Compute | Tile Size |
|:---|:---|:---:|:---|:---|
| **A — Direct** | `< MSTS_DIRECT_MAX` (≈3 MB) | 0 | Single AVX loop | Full tensor |
| **B — Single-thread** | `< 32 MB` | 1 | Inline main thread | ≈75% L2 (192 KB) |
| **C — Full CrookScheduler** | `≥ 32 MB` | 2 | `rayon` parallel | 4 MB (SATA burst) |

Thresholds are derived from `build.rs` reading L2/L3 sysfs or compile-time env vars.

### Python API

```python
import vulkannn_rusted as vnn

# Write RAM tensor to SSD (Path A/B/C auto-selected)
t = vnn.Tensor(data=my_array, dtype=vnn.DataType.F32, device="cpu")
ssd_t = t.save_ssd("/tmp/weights.bin")

# Apply in-place MSTS operation and read back as f32
result_ssd = ssd_t.unary_op_ssd("relu", 0.0, 0.0)
result_f32  = result_ssd.load_to_f32_vec_msts()  # Vec<f32> → Python list
```

---

## Streaming Module

Source: `src/streaming.rs`

`BUDGETS` stores `l2_ram_max_bytes`: threshold below which the full tensor loads into RAM before computation. Above this, the MSTS SSD streaming path activates.
