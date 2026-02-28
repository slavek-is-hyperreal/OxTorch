# 📖 VNN Technical Manual: Source Code Walkthrough

This document provides a comprehensive, block-by-block (and often line-by-line) explanation of the VNN Dual-Engine Architecture. It covers both the ultra-performant Native Rust Engine and the legacy Python computational engine.

---

## 🦀 Rusted Engine (`vulkannn_rusted/`) 
The modern core. Exposes pure, memory-safe, blazing fast C/Rust native objects to Python via PyO3.

### 1. [tensor.rs](../vulkannn_rusted/src/tensor.rs) - The Native Wrapper
This module binds Python objects to Rust raw memory constructs.
* **`#[pyclass] struct Tensor`**: The object seen when typing `vnn.Tensor()` in Python. It contains the shape, the device mapping, and the `mmap_data` which holds the zero-copy Arc pointer to the active NVMe SSD file slice.
* **SIMD & Rayon (`device="cpu"`)**: When an addition or MatMul is invoked with the CPU device, `tensor.rs` bypasses the Python interpreter completely. It invokes `rayon` thread pools (`.par_iter_mut()`), saturating host cores with SIMD ops.
* **BLAS Routing (`super::matrixmultiply`)**: Matrix multiplications jump into highly tuned, hand-written assembly unrolled registers.

### 2. [backend.rs](../vulkannn_rusted/src/backend.rs) - WGPU Pipeline
This handles the GPU interface, completely sidestepping Python's Taichi module.
* **`WgpuBackend` Initialization**: Binds to Vulkan (or Metal/DX12 on Mac/Windows). Sets up Global Device handles and Memory Hints pointing towards high-performance DMA buffers.
* **`WGSL` Complied Modules**: Compute shaders (like `add.wgsl` or `matmul.wgsl`) are attached as compiled `ComputePipeline` instances. No JIT stall time is present here. 
* **True Heterogeneous Routing (`execute_add / execute_matmul`)**: If `is_hybrid` is active, the function explicitly forks background workers using `std::thread::scope`. The GPU threads submit mapped chunk slices to `WGPU Queue` Ping-Pong buffers, while CPU threads fall back to `Rayon` computations on the RAM layer. They converge synchronously.

### 3. [streaming.rs](../vulkannn_rusted/src/streaming.rs) - Extreme Pipelining
The system enforcing the "VRAM is a Cache" ruleset.
* **`L3Cache::map_ssd_tensor`**: Returns a pointer to raw `memmap2` regions on disk. Zero allocation is used for gigabytes of data.
* **`BUDGETS` LazyLock**: Tracks `l1_vram_max_bytes` and `l2_ram_max_bytes` using `AtomicUsize`. If a WGPU calculation exceeds VRAM capacity constraints, it is tiled via chunk offsets preventing driver timeouts and graphical OS freezes.

---

## 🏗️ Legacy Engine (`vulkan_nn_lib/`)

### 1. [tensor.py](../vulkan_nn_lib/tensor.py) - The Universal Tensor
This is the central object of VNN Legacy. It manages the state, device routing, and Autograd graph.

*   **Lines 16-21**: `setup_ssd_storage` initializes the `TensorStore` globally.
*   **Lines 34-191**: `__init__` constructor.
    *   It handles **Auto-Device Selection**: decides whether a tensor should live in VRAM, RAM, or SSD based on its size and the `MemoryManager` budget. (Lines 76-88).
    *   **int4 Packing**: (Lines 114-118, 137-138) Uses a specialized packing/unpacking logic for 4-bit weights.
*   **Lines 24-26**: `from_numpy` zero-copy factory.
*   **Lines 257-285**: `_acc_grad()` - The most critical part for OOM-safety. 
    *   It accumulates gradients on the **target device**. 
    *   If a parameter is on SSD, its gradient is also on SSD, and they are summed using the streaming engine.
*   **Lines 286-310**: `backward()` implementation.
    *   It builds a **topological sort** of the computation graph using a depth-first search (DFS).
*   **Lines 318-343**: `to_numpy()` - Standardized data retrieval with a safety fallback for Vulkan data currently on CPU.
*   **Lines 475-508**: **Activation Methods**. Direct support for `relu`, `silu`, `leaky_relu`, `softmax`, and `gelu_tanh` on the `Tensor` object.
*   **Lines 510-545**: **MatMul Implementation**. Detects CPU-resident data and uses PyTorch's optimized BLAS kernels via shared memory views.

### 2. [streaming_ops.py](../vulkan_nn_lib/streaming_ops.py) - The ARAS Engine
The "Heart" of VNN's OOM-safety. It implements tiled streaming for multi-gigabyte tensors.

*   **Lines 30-128**: `TilePrefetcher` - An asynchronous producer-consumer queue.
    *   **Bounded Queue**: (Line 41) Now uses a **dynamic queue** to provide backpressure, preventing disk prefetching from overflowing RAM on "Monster-Scale" datasets.
    *   **Adaptive Backoff**: (Lines 59-81) Monitors RAM usage and scales the number of outstanding I/O futures based on the safe budget.
*   **Lines 739-805**: `SOE.sum()` (and other reductions).
    *   Uses a **ThreadPoolExecutor** with **Future-based throttling** (Lines 782-789) to ensure compute doesn't lag too far behind I/O.
    *   **Lines 800-805**: `SafetyViolationError` catch-all. If RAM spikes dangerously, it triggers an **Adaptive Restart**, reducing the budget and retrying the operation.
*   **Lines 144-518**: `elementwise_op()`.
    *   Implements **Heterogeneous Acceleration**. It can run in **Hybrid Mode**, where one GPU thread processes tiles in Vulkan while CPU threads handle other tiles in parallel. (Lines 465-501).
    *   **Kaggle Redirect**: (Lines 167-172) Automatically offloads operations exceeding threshold to cloud GPUs.
    *   **Activation Integration**: Native support for streaming activation functions directly on disk.

### 3. [kernels.py](../vulkan_nn_lib/kernels.py) - Taichi Compute Shaders
JIT-compiled SPIR-V shaders for Vulkan.

*   **Lines 10-15**: Taichi initialization forcing Vulkan backend.
*   **Lines 22-44**: Adam/SGD kernels. These allow weight updates to happen entirely on the GPU.
*   **Lines 497-501**: `k_reduce_sum`. 
    *   Uses an **f64 accumulator** to prevent precision loss when summing billions of elements.
*   **Kernel Signature Standards**: All math kernels now follow the `(Input, Output, Total)` out-of-place pattern to ensure thread-safety and avoid race conditions in tiled execution.
*   **Lines 504-512**: `k_unpack_int4`. 
    *   Performs bit-shifting and masking to decompress two 4-bit weights from one 8-bit byte on the fly.

### 4. [memory.py](../vulkan_nn_lib/memory.py) & [memory_pool.py](../vulkan_nn_lib/memory_pool.py) - Hardware Oracles & Allocators
Manages the RAM/VRAM budgets and low-level physical allocations.

*   `memory.py`:
    *   **Line 15**: `_system_reserve_bytes`. Can be set via environment variable `VNN_RESERVE_GB` (Default: 5). Protects ZFS ARC/system background limits.
    *   **Lines 111-125**: `get_safe_budget()`. Dynamically calculates 80% of **(Available RAM - Reservation)**.
*   `memory_pool.py`:
    *   **VulkanTensorPool**: Suballocator wrapper around `ti.ndarray` to constrain total allocation calls matching Vulkan driver caps.

### 4.5. [paged_attention.py](../vulkan_nn_lib/paged_attention.py) - Virt-to-Phys Memory
Implements the Phase 2 KV cache virtualization structure for LLM decoding contexts.

*   `KVCachePool`: Real pre-allocated memory pool on Vulkan arrays (`physical_k`, `physical_v`).
*   `BlockTable`: Maps a logical sequence to random scattered blocks.
*   `PagedKVCache`: Context manager injected into the LLM layer simulating contiguous structures while utilizing `BlockTable` internally.

### 5. [kaggle_executor.py](../vulkan_nn_lib/kaggle_executor.py) - Ephemeral Supercompute
The orchestration engine for remote cloud compute.

- **Lines 140-192**: Job submission and script generation.
- **Lines 232-282**: Remote kernel polling and result streaming.

---

## 🧠 Brain & Interface

### 5. [torch_shim.py](../vulkan_nn_lib/torch_shim.py) - The Compatibility Layer
Provides the `torch.*` API.

*   **Lines 63-131**: `zeros()`, `randn()`, `ones()` - These are "Smart Factories".
    *   They check if the requested shape will exceed RAM *before* allocating anything.
    *   If it exceeds RAM, they return an **SSD-resident tensor** immediately.
*   **Lines 137-139**: `from_binary()`. 
    *   Mounts a file as an SSD tensor without reading a single byte (Zero-copy).

### 6. [optimizers.py](../vulkan_nn_lib/optimizers.py) - Training Core
*   **Lines 169-298**: `AutoAdam`.
    *   The most advanced optimizer. It segments parameters: some in VRAM, some in RAM, some on SSD. 
    *   Updates are orchestrated to never overflow any bucket.

---

## 📦 Data Storage

### 7. [tensor_store.py](../vulkan_nn_lib/tensor_store.py) - SSD Persistence
*   **Lines 21-44**: Uses **numpy memmap** with `mode='r+'`.
    *   Allows VNN to treat the SSD as virtual memory, delegating caching to the Linux kernel / ZFS ARC.

---

## 🎭 High-Level Modules ([vulkan_nn_lib/modules/](../vulkan_nn_lib/modules/))

### 8. [layers.py](../vulkan_nn_lib/modules/layers.py)
*   **Linear**: Detects if weights are too large for RAM and initializes them on SSD.
*   **RMSNorm**: Optimized Taichi kernel for stable normalization.
*   **PagedAttention**: High-efficiency VRAM virtualization bypassing contiguous caching. Calls `.get_physical_blocks()` on the supplied `PagedKVCache` to submit raw offset vectors directly to the custom Taichi Vulkan pipeline.

### 9. [models.py](../vulkan_nn_lib/modules/models.py)
*   **Gemma3Block**: Implements the complex Gemma-3 architecture (AltUp, Laurel, PLE) while keeping activation buffers OOM-safe via the core engine.

---
*This manual is updated dynamically to reflect the latest architectural refinements in VNN.*
