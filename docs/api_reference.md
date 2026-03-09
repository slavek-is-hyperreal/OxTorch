# 📖 API Reference (v3.2.0 "Valkyrie")

This document provides a detailed reference for the `vulkannn_rusted` engine.

---

## 🏗 `DataType` Enum
*   **Source**: `src/tensor.rs:16`
*   **Variants**:
    - `DataType.F32`: Standard 32-bit floating point.
    - `DataType.F16`: 16-bit half-precision floating point.
    - `DataType.BF16`: Brain Floating Point (8-bit exponent, 7-bit mantissa).

---

## 🏛 `Tensor` Class
*   **Source**: `src/tensor.rs:31`
*   **Constructor**: `Tensor(data=None, shape=None, dtype=DataType.F32, device="auto", name="Tensor")` (`src/tensor.rs:49`)
    - `data`: Optional `numpy.ndarray` (automatically cast if needed).
    - `shape`: Optional `tuple` of dimensions.
    - `dtype`: `DataType` variant. Default is F32.
    - `device`: `"cpu"`, `"vulkan"`, `"hybrid"`, or `"auto"`.
    - `name`: Human-readable label for debugging (`src/tensor.rs:37`).

### Static Methods
1.  **`from_ssd(path: str, shape: list, dtype=DataType.F32)`** (`src/tensor.rs:83`)
    - Maps an existing binary file to a Tensor via `memmap2`.
    - **Performance**: Uses `libc::madvise(MADV_SEQUENTIAL)` (`src/tensor.rs:92`) for hardware prefetching.
    - Returns a `ReadOnly` tensor.
2.  **`new_ssd(path: str, shape: list, dtype=DataType.F32)`** (`src/tensor.rs:98`)
    - Creates a new file on disk and maps it as `ReadWrite` (`src/tensor.rs:103`).
    - Ideal for storing results of massive calculations (e.g. 16GB ReLU).

### Matrix Operations
*   **`__matmul__(other: Tensor)`** (`src/tensor.rs:260`)
    - Operator: `@`
    - **Hybrid Dispatch**: If `device="hybrid"`, uses dynamic work-stealing (`src/backend.rs:388`).
    - **CPU Path**: Uses `matrixmultiply::sgemm` (`src/tensor.rs:639`).
    - **GPU Path**: Uses WGSL tiling shader (`src/shaders/matmul.wgsl`).
    - **Fallbacks**: Automatically casts F16/BF16 to F32 for GPU compute if hardware doesn't support native FP16 (`src/backend.rs:344`).

### Activation Functions
*   **`relu()`** (`src/tensor.rs:216`) -> Uses `execute_activation` (`src/backend.rs:583`).
*   **`sigmoid()`** (`src/tensor.rs:256`)
*   **`silu()`** (`src/tensor.rs:257`)
*   **`*_into(out: Tensor)`**: Result-streaming variants (e.g., `relu_into`).
    - **Logic**: Directly writes to the `out` tensor to avoid allocations.
    - **Thresholds**: Small tensors (<2M elements) are processed serially on CPU to avoid Vulkan overhead (`src/backend.rs:597`).

### Hardware Interop
*   **`to_numpy()`** (`src/tensor.rs:108`)
    - Parallel conversion from Rust types (F16/BF16) to NumPy F32 via Rayon (`src/tensor.rs:116`).
*   **`device` (property)**: Switch backends at runtime.

---

## ⚙️ Backend Module
*   **Initialization**: `init_backend()` (`src/backend.rs:28`) - Configures the `WgpuBackend` singleton with `HighPerformance` preference.
*   **Buffer Caching**: Prevents `wgpu` stalls via `get_buffer` (`src/backend.rs:99`) and `recycle_buffer` (`src/backend.rs:110`).
*   **Tiling Config**: `src/backend.rs:360`
    - `CPU/GPU Split`: 30% CPU / 70% GPU (default in Hybrid mode).
    - `Tile Size`: 512x16384 (Large MatMul) or 16x2048 (GEMV/Small).

---

## 🎨 Shaders
*   **MatMul WGSL**: `src/shaders/matmul.wgsl`
    - Uses 16x16 workgroup tiles.
    - Implements **Workgroup Shared Memory (L1)** caching for `tile_a` and `tile_b` (`matmul.wgsl:14-15`) to maximize TFLOPS.
