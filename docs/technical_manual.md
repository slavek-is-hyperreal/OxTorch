# 📖 VNN Technical Manual: Source Code Walkthrough

This document provides a comprehensive, block-by-block (and often line-by-line) explanation of the VNN Legacy Edition codebase. It is designed to help developers understand the "Why" and "How" behind every critical function.

---

## 🏗️ Core Engine (`vulkan_nn_lib/`)

### 1. [tensor.py](../vulkan_nn_lib/tensor.py) - The Universal Tensor
This is the central object of VNN. It manages the state, device routing, and Autograd graph.

*   **Lines 14-16**: `setup_ssd_storage` initializes the `TensorStore` globally.
*   **Lines 34-195**: `__init__` constructor.
    *   It handles **Auto-Device Selection**: decides whether a tensor should live in VRAM, RAM, or SSD based on its size and the `MemoryManager` budget.
    *   **int4 Packing**: (Lines 140-145) Uses a **+8 bias** when packing inputs to `uint8` to ensure correct round-tripping with the dequantization logic.
*   **Lines 20-32**: Factory-style logic for zero-copy and RAM awareness.
*   **Lines 258-281**: `_acc_grad()` - The most critical part for OOM-safety. 
    *   It accumulates gradients on the **target device**. 
    *   If a parameter is on SSD, its gradient is also on SSD, and they are summed using the streaming engine.
*   **Lines 283-306**: `backward()` implementation.
    *   It builds a **topological sort** of the computation graph using a depth-first search (DFS).
*   **Lines 308-339| **I/O Strategy** | Standard OS Memmap | Greedy ARAS Buffering (Linear Scaling)|
*   **Lines 308-339**: `to_numpy()` - Standardized data retrieval with a safety fallback for Vulkan data currently on CPU.
*   **Lines 469-502**: **Activation Methods**. Direct support for `relu`, `silu`, `leaky_relu`, `softmax`, and `gelu_tanh` on the `Tensor` object.
*   **Lines 504-534**: **MatMul Implementation**. Detects CPU-resident data and uses PyTorch's optimized BLAS kernels via shared memory views.

### 2. [streaming_ops.py](../vulkan_nn_lib/streaming_ops.py) - The ARAS Engine
The "Heart" of VNN's OOM-safety. It implements tiled streaming for multi-gigabyte tensors.

*   **Lines 21-110**: `TilePrefetcher` - An asynchronous producer-consumer queue.
    *   **Bounded Queue**: (Line 32) Now uses a **fixed-size queue** to provide backpressure, preventing disk prefetching from overflowing RAM on "Monster-Scale" datasets.
    *   **Adaptive Backoff**: (Lines 43-60) Monitors RAM usage and scales the number of outstanding I/O futures based on the safe budget.
*   **Lines 522-589**: `SOE.sum()` (and other reductions).
    *   Uses a **ThreadPoolExecutor** with **Future-based throttling** (Lines 544-558) to ensure compute doesn't lag too far behind I/O.
    *   **Lines 530-545**: `SafetyViolationError` catch-all. If RAM spikes dangerously, it triggers an **Adaptive Restart**, reducing the budget and retrying the operation.
*   **Lines 125-324**: `elementwise_op()`.
    *   Implements **Heterogeneous Acceleration**. It can run in **Hybrid Mode**, where one GPU thread processes tiles in Vulkan while CPU threads handle other tiles in parallel.
    *   **Dtype Promotion**: Automatically promotes integer division to `float32` for PyTorch parity.
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

### 4. [memory.py](../vulkan_nn_lib/memory.py) - Hardware Oracle
Manages the RAM/VRAM budgets and detects safety violations.

*   **Line 15**: `_system_reserve_bytes`. Can be set via environment variable `VNN_RESERVE_GB` (Default: 5). This memory is "hidden" from VNN to protect ZFS ARC or other background services.
*   **Lines 90-105**: `get_safe_budget()`. 
    *   **Linear Scaling**: Dynamically calculates 80% of **(Available RAM - Reservation)**. 
    *   **No Hard Cap**: Scales naturally from 1GB to 1TB+. On a 128GB system, VNN will automatically utilize ~100GB of RAM before offloading to SSD.
*   **Lines 139-153**: `wait_for_ram()`. 
    *   Blocking check used by factory functions. If RAM is too low, it pauses the requester until memory is released.

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

### 9. [models.py](../vulkan_nn_lib/modules/models.py)
*   **Gemma3Block**: Implements the complex Gemma-3 architecture (AltUp, Laurel, PLE) while keeping activation buffers OOM-safe via the core engine.
 bitumen
