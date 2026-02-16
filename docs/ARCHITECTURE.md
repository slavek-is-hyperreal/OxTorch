# VulkanNN Architecture Deep Dive

This document details the internal architecture of **VulkanNN**, a lightweight PyTorch-like library designed to run large language models (like Gemma 3n) on consumer-grade GPUs via the Vulkan API.

## 1. Core Philosophy: RAM-Centric Computing

The primary goal of VulkanNN is **accessibility** and **scalability on legacy/budget hardware**. 

### VRAM is a Cache, RAM is the Mainframe
Unlike traditional GPU frameworks that fail when VRAM is full, VulkanNN treats the GPU as a **computational accelerator with a large cache (VRAM)**.
-   **RAM is the Source of Truth**: All model weights, gradients, and optimizer states (Adam's `m` and `v`) reside in high-capacity system RAM (DDR3/DDR4/DDR5).
-   **VRAM is a Streaming Buffer**: Weights are paged into VRAM for active computation and immediately flushed back or overwritten.
-   **Scalability**: This allows a card with only **2GB VRAM** to handle models of **128GB+** size, provided enough system RAM is available. It is slower than CUDA, but it **works** where others crash.

## 2. Directory Structure

```
vulkan_nn_lib/
├── core.py          # Legacy entry point (imports from others)
├── tensor.py        # The `Tensor` class - heart of the engine
├── kernels.py       # Raw Taichi kernels (compute shaders)
├── modules/         # Neural network layers (replicating torch.nn)
│   ├── base.py      # Base Module class
│   ├── layers.py    # Standard layers (Linear, RMSNorm, Embedding)
│   └── tiled.py     # TiledLinear (RAM-VRAM streaming)
├── functional.py    # Activation functions (ReLU, Softmax, etc.)
└── torch_shim.py    # Emulates `torch` namespace for drop-in compatibility
```

## 3. The `Tensor` Class (`tensor.py`)

The `Tensor` class is a wrapper around `ti.ndarray` (Taichi's GPU array). It handles:

1.  **Device Abstraction**:
    -   `device='vulkan'`: Data resides in VRAM (`ti.ndarray`).
    -   `device='cpu'` / `'ram'`: Data resides in system RAM (`np.ndarray`).
2.  **Autograd Tracking**:
    -   Each `Tensor` stores a `grad` (gradient tensor) and a `_backward_fn`.
    -   `requires_grad=True` enables graph building.
3.  **Synchronization**:
    -   Operations on Vulkan are asynchronous.
    -   Methods like `to_numpy()` or `load_from_numpy()` automatically call `ti.sync()` to ensure data consistency.

### Key Mechanism: The Backward Graph
When an operation (e.g., `C = A + B`) is performed:
1.  A new `Tensor` C is created.
2.  A closure `_backward()` is defined that calculates `grad_A` and `grad_B` using `grad_C`.
3.  C stores this closure in `_backward_fn` and adds A and B to its `_prev` set.
4.  Calling `C.backward()` triggers a topological sort of the graph and executes these closures in reverse order.

## 4. Compute Kernels (`kernels.py`)

All heavy lifting is done by **Taichi kernels**. These are Just-In-Time (JIT) compiled to SPIR-V (Vulkan compute shaders).

-   **Design**: Kernels operate on 1D flattened arrays to avoid complex shape handling in shader code.
-   **Parallelization**: Loops are automatically parallelized by Taichi.
-   **Optimization**: We avoid `atomic_add` where possible (e.g., in TiledLinear gradients) to prevent race conditions and improve speed.

## 5. Tiled Linear Layers (`modules/tiled.py`)

This is the **critical innovation** that allows running 4B+ parameter models on 2GB VRAM cards.

### The Problem
A standard linear layer $Y = XW^T$ requires storing the entire weight matrix $W$ in VRAM. For Gemma 3n 4B, this is several gigabytes, crashing small GPUs.

### The Solution: Tiling
`TiledLinear` keeps the main weights in **system RAM**. During the forward pass:
1.  It allocates a small "tile" buffer in VRAM (e.g., 32MB).
2.  It loops over the weight matrix in chunks.
3.  **Copy**: Uploads a chunk of $W$ from RAM to the VRAM tile.
4.  **Compute**: Performs matrix multiplication for that chunk, accumulating results into the output buffer.
5.  **Repeat**: Moves to the next chunk.

This allows technically *infinite* model size support, limited only by system RAM and PCIe bandwidth.

**Backward Pass**:
The same logic applies to gradients. We compute gradients for a tile, download them to RAM, and accumulate them into the master weight gradients in RAM, keeping VRAM usage constant.

## 6. Autograd Engine

VulkanNN implements a custom Reverse-Mode Automatic Differentiation engine.

-   **Graph Construction**: Dynamic (Run-by-Run), similar to PyTorch.
-   **Supported Ops**: Add, Mul, MatMul, Softmax, RMSNorm, RoPE, Silu, Gelu, Embedding.
-   **Validation**: All gradients contain unit tests verifying them against PyTorch's Autograd to `1e-5` tolerance.

## 7. Hardware Calibration & Tuning

Since VulkanNN treats **VRAM as a Cache**, performance depends on finding the "Sweet Spot" for your specific card and PCIe bandwidth.

### The `tile_size` parameter
The `tile_size` determines how much data is moved from RAM to GPU in a single "packet".

-   **Too Small**: High overhead from many small PCIe transfers.
-   **Too Large**: Out Of Memory (OOM) errors, or system sluggishness if you starve the display driver.

### The "Fast BAR" Threshold
Most GPUs (especially older ones like the R7 260X) have a **Visible VRAM BAR** (usually 256MB). Data within this window can be accessed by the CPU at full speed.
-   **Optimization Tip**: Set your optimizer `tile_size` so that all four Adam caches (`p`, `g`, `m`, `v`) fit into this 256MB window.
-   **Example**: `tile_size = 16 * 1024 * 1024` (exactly 64MB per cache, total 256MB).

### Recommendation Table

| GPU Tier | VRAM | Recommended BAR/Tile Size | Strategy |
| :--- | :---: | :--- | :--- |
| **Legacy (R7 260X)** | 1-2GB | **64MB** (16M elements) | RAM-Centric (everything paged) |
| **Mid-Range (RX 580)** | 8GB | **256MB** (64M elements) | Hybrid (Nano models resident) |
| **High-End (RTX 3090)** | 24GB+ | **1GB+** (256M elements) | Full-VRAM preferred |

## 8. Performance Tips

-   **Batching**: Always use batch sizes > 1 to amortize kernel launch overhead.
-   **Reuse Tensors**: Avoid creating new tensors in loops; use `zero_()` if possible (future optimization).
-   **VRAM Management**: If you get OOM (Out Of Memory), reduce the `tile_size` in `TiledLinear`.

## 8. Contributing

-   **New Layers**: Add usage of `kernels.py` functions in `modules/layers.py` and register backward functions.
-   **New Kernels**: Add `@ti.kernel` functions in `kernels.py`. Remember to update `kernels.py` to support new operations.

---
*Maintained by the Tensor Forever Team.*
