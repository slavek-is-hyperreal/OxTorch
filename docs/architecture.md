# OxTorch Architecture: Deep Dive

This document provides a technical overview of the OxTorch engine, from the Rust source code structure to the SSD streaming mechanisms and the Vulkan backend.

---

## 1. Source Map (Rust)

The core logic of OxTorch resides in the `vulkannn_rusted/src/` directory:

```text
src/
├── lib.rs              # Python module entry point (PyO3)
├── tensor/             # Core Tensor logic
│   ├── mod.rs          # PyO3 methods, constructors, fallback dispatch
│   ├── storage.rs      # Aligned memory management, AlignedBuffer
│   ├── ops.rs          # Operation trait and dispatch bridge
│   ├── msts.rs         # MSTS: 3-path SSD streaming logic
│   └── pool.rs         # TensorPool: Slab allocator
├── cpu/                # CPU Backend
│   ├── mod.rs          # CPU backend initialization
│   └── ops/            # specialized kernels (avx, neon, swar)
└── vulkan/             # Vulkan Backend (ash)
    ├── mod.rs          # Vulkan instance/device/queue setup
    ├── backend.rs      # Shader dispatch, descriptor pools, command buffers
    └── shaders/        # GLSL/WGSL source code
```

---

## 2. Python Layer (`oxtorch/`)

The high-level Python package acts as a drop-in replacement for PyTorch:

*   **`oxtorch/__init__.py`**: Module-level `__getattr__` for global functions (e.g., `torch.add`).
*   **`oxtorch/tensor.py`**: The `Tensor` class wrapper. It handles properties (`shape`, `dtype`, `device`) and manages the dynamic fallback to PyTorch if no native kernel exists.
*   **Fallback Logic**: If `device="ssd"`, it uses `msts_pytorch_apply` (tiled fallback). Otherwise, it pulls data to RAM and executes via regular PyTorch.

---

## 3. Vulkan Backend (`ash`)

OxTorch uses a direct implementation via `ash` (Vulkan 1.2), bypassing high-level abstraction layers for maximum performance on older GPUs (mobile and desktop).

*   **Timeline Semaphores**: Used for lockless synchronization between the CPU and GPU.
*   **Descriptor Set Pooling**: Aggressive caching of descriptor sets to avoid driver-level allocation overhead during inference.
*   **Direct Memory Access**: Buffers are mapped directly into host memory where possible (Vulkan `HOST_VISIBLE | HOST_COHERENT`).

---

## 4. MSTS: Mera Style Tiling System

MSTS is a unique orchestration layer that enables processing models larger than RAM (Out-of-Core).

*   **3-Path Dispatch**:
    1.  **Direct**: No threading overhead for tiny tensors.
    2.  **Single-thread**: Sequential tile processing for medium tensors.
    3.  **Full Parallel (CrookScheduler)**: Asynchronous "Race for Tiles" using `io_uring` and `rayon`.
*   **Tiling logic**: Tensors are split into 1MB-4MB tiles adjusted to the CPU's cache size.

---

## 5. Memory Management: TensorPool

To avoid the overhead of `malloc`/`free` during inference, OxTorch uses a custom slab allocator:

*   **Thread-Local**: Each thread has its own pool (zero contention).
*   **6-Bucket Strategy**: Manages buffers from 4KB up to 256MB+.
*   **Zero-Copy Integration**: Directly feeds buffers to MSTS and CPU kernels.

---

## 6. SSD Streaming (`io_uring`)

On Linux, OxTorch utilizes `io_uring` + `O_DIRECT` for high-throughput SSD access.

*   **Bypassing Page Cache**: Eliminates double buffering and provides deterministic performance.
*   **Kernel-side Async I/O**: Allows the CPU to focus on computation while the kernel and NVMe controller manage data transfer in the background.
