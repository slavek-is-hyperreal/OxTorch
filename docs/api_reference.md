# 📖 API Reference (v2.9.0)

This document provides a detailed reference for the `vulkannn_rusted` Python class.

---

## `Tensor` Class
*   **Source**: `src/tensor.rs:14`
*   **Constructor**: `Tensor(data=None, shape=None, device="auto", name="Tensor")` (`src/tensor.rs:28`)
    - `data`: Optional - `numpy.ndarray` (f32).
    - `shape`: Optional - `tuple` or `list` of dimensions.
    - `device`: `"cpu"`, `"vulkan"`, `"hybrid"`, or `"auto"`.
    - `name`: Human-readable label for debugging (`src/tensor.rs:20`).

### Static Methods
1.  **`from_ssd(path: str, shape: list)`** (`src/tensor.rs:43`)
    - Maps an existing binary file (f32) to a Tensor via `memmap2`.
    - Automatically hints the kernel with `MADV_SEQUENTIAL`.
    - Returns a `ReadOnly` tensor.
2.  **`new_ssd(path: str, shape: list)`** (`src/tensor.rs:56`)
    - Creates a new binary file on disk and maps it as `ReadWrite`.
    - Used for storing large results (e.g., 20GB+) that exceed RAM.

### Matrix Operations
*   **`__matmul__(other: Tensor)`** (`src/tensor.rs:98`)
    - Operator: `@`
    - Logic: Depending on `device`, it triggers the Vulkan tiling engine (`src/backend.rs:262`) or the CPU sgemm engine (`src/tensor.rs:228`).
    - Support: 2D matrices (MatMul) and 1D vectors (GEMV).

### Element-wise Operations
*   **`__add__(other: Tensor)`** (`src/tensor.rs:73`)
    - Operator: `+`
    - High-performance parallel addition via Rayon (CPU) or WGSL (Vulkan).
*   **`relu()`** (`src/tensor.rs:88`)
    - Standard Rectified Linear Unit.
*   **`sigmoid()`** (`src/tensor.rs:94`)
*   **`silu()`** (`src/tensor.rs:95`)
    - Sigmoid Linear Unit (used in Gemma/LLama models).

### Utility Methods
*   **`to_numpy()`** (`src/tensor.rs:65`)
    - Converts the Tensor back to a `numpy.ndarray`.
    - Note: This involves a copy from GPU/SSD to RAM if not already there.
*   **`device` (property)**
    - Get/Set the execution device (`cpu`, `vulkan`, `hybrid`). Change this at runtime to switch backends.

---

## ⚙️ Backend Module
*   **Initialization**: `vulkannn_rusted.backend.init_backend()` (`src/lib.rs:17`) - done automatically on module import.
*   **Buffer Caching**: `src/backend.rs:95` (`get_buffer`) and `src/backend.rs:106` (`recycle_buffer`). This prevents high-latency `wgpu::Device::create_buffer` calls during tight loops.
*   **Work Stealer**: `src/backend.rs:279` implements the Atomic Fetch-Add logic to divide M-blocks between workers.
