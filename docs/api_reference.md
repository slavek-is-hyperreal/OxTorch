# VulkanNN Rusted - API Reference (v3.6.0)

This document provides a technical overview of the `vulkannn_rusted_dev` library.

---

## DataType Enum

Source: `src/tensor.rs:19`

| Variant | Description |
|:---|:---|
| `DataType.F32` | 32-bit IEEE 754 float |
| `DataType.F16` | 16-bit IEEE 754 half-precision float |
| `DataType.BF16` | Brain Float 16 (8-bit exponent, 7-bit mantissa) |
| `DataType.Int8` | 8-bit signed integer |

---

## Tensor Class

Source: `src/tensor.rs:53`

### Constructor

```python
Tensor(data=None, shape=None, dtype=DataType.F32, device="auto", name="Tensor")
```

- `data`: Optional `numpy.ndarray` (F32). Converted to the target dtype at construction.
- `shape`: Alternative to data. Allocates a zero-filled tensor of the given shape.
- `dtype`: Storage precision. Conversion uses runtime-dispatched SIMD (see avx_swar.rs).
- `device`: `"cpu"`, `"vulkan"`, `"hybrid"`, or `"auto"` (defaults to CPU).
- `name`: Label used in debug output.

### Static Methods

**`Tensor.from_ssd(path, shape, dtype=DataType.F32)`** — `src/tensor.rs:149`

Maps an existing binary file as a read-only SSD tensor backed by `io_uring`/`O_DIRECT`.
The file must already exist and be at least `prod(shape) * bytes_per_element` bytes.
Returns a tensor with `device="ssd"` and no in-memory storage.

**`Tensor.new_ssd(path, shape, dtype=DataType.F32)`** — `src/tensor.rs:242`

Creates a new file on disk and maps it read-write. Used for out-of-core results
(e.g., the 16GB Monster ReLU benchmark). Aligned to 1MB ZFS recordsize boundaries.

---

### Matrix Operations

**`__matmul__(other: Tensor)`** — `src/tensor.rs:747` — operator: `@`

Dispatches based on `device`:

- `"cpu"`: Uses `matrixmultiply::sgemm` (F32) or SWAR upcast + sgemm (F16/BF16).
- `"vulkan"`: Uploads A and B to GPU via staging buffers, runs SPIR-V compute shader,
  downloads result. All types computed as F32 on GPU (Bonaire has no native F16 math).
- `"hybrid"`: Currently routes to Vulkan for MatMul (tile-pulling Phase 4 applies to
  activations only; MatMul hybrid tiling is in progress).
- `"ssd"`: Loads tiles via `load_to_f32_vec_msts()` per the MSTS ring buffer, then runs sgemm.

---

### Activation Functions

| Method | Source line | Notes |
|:---|:---|:---|
| `relu()` | `src/tensor.rs:660` | Returns new Tensor |
| `sigmoid()` | `src/tensor.rs:742` | Returns new Tensor |
| `silu()` | `src/tensor.rs:743` | Returns new Tensor |
| `relu_into(out)` | `src/tensor.rs:661` | Writes to pre-allocated out Tensor |
| `sigmoid_into(out)` | `src/tensor.rs:662` | Writes to pre-allocated out Tensor |
| `silu_into(out)` | `src/tensor.rs:663` | Writes to pre-allocated out Tensor |

For `device="hybrid"`, activations use the MSTS tile-pulling dispatch:

- Tensors >= 4M elements: one GPU thread + one CPU SWAR thread race for tiles.
- Tensors < 4M elements: GPU dispatcher is skipped. Pure CPU SWAR (Bonaire PCIe cost ~80ms).

---

### Other Methods

**`to_numpy()`** — `src/tensor.rs` — Converts F16/BF16 storage back to F32 numpy array
via Rayon parallel SIMD upcast.

**`transpose()`** — Returns a transposed view (no data copy, flips `is_transposed` flag
and stride parameters used in sgemm dispatch).

---

## Backend Module

Source: `src/backend.rs`

### Initialization

`init_backend()` creates the global `BACKEND: OnceLock<AshBackend>` singleton using Vulkan 1.2.
Called once automatically when the Python module is imported via `lib.rs`.

### Key Public Functions

| Function | Description |
|:---|:---|
| `execute_activation(input, op, dtype, is_hybrid)` | Full-tensor GPU activation |
| `execute_activation_chunked(input, output, offset, count, op, dtype)` | Tile-range GPU activation for MSTS dispatch |
| `execute_activation_into(input, op, output, dtype, is_hybrid, staging)` | Sync activation with in-place output |
| `execute_matmul(a, b, m, k, n, dtype, is_hybrid)` | Full matrix multiply on GPU |
| `execute_add_into(a, b, out, dtype, is_hybrid, staging)` | Element-wise addition |
| `poll_async_ops()` | Recycles completed GPU operations |
| `poll_async_ops_until(wait_id)` | Blocks until a specific timeline semaphore value is reached |

### Buffer Cache

`get_buffer(size, usage, label, cpu_visible)` — returns a `CachedBuffer` from the recycle pool
or allocates a new one via `gpu_allocator`. CPU-visible (staging) buffers use `CpuToGpu` memory;
device buffers use `GpuOnly` to avoid coherency issues on legacy AMD (Bonaire).

`recycle_buffer(buf)` — returns a buffer to the cache for reuse.

---

## Streaming Module

Source: `src/streaming.rs`

Manages RAM budget detection, prefetcher initialization, and the background SSD prefetch thread.

`BUDGETS` stores `l2_ram_max_bytes`: the threshold below which the full tensor will be loaded
into RAM before computation. Above this, the MSTS streaming path is used.

---

## MSTS Scheduler

Source: `src/crook_scheduler.rs`

`StatefulTile` — a 1MB-aligned structure with an `AtomicU32` status field representing tagged
dataflow states (EMPTY -> LOADING -> READY_CPU -> READY_GPU -> GPU_COMPUTING -> GPU_DONE).
Workers poll and claim tiles via Compare-And-Swap without any mutex.
