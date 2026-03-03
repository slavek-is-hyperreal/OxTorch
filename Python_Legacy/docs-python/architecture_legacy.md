# Architecture: VNN Legacy Engine (Python + Taichi)

> [!IMPORTANT]
> This document describes the legacy Python architecture. For the modern high-performance Rust engine, see [docs/architecture.md](../../docs/architecture.md).

## 1. The Legacy Technical Stack
The original VNN engine (`import vulkan_nn_lib`) is a Python-first implementation designed for maximum flexibility.
- **Backend**: Taichi JIT-compiled SPIR-V Vulkan shaders.
- **Streaming**: Python-based ARAS/SOE engines manage data tiling.
- **Memory Management**: Uses `VulkanTensorPool` (Slab allocator) in Python to bypass driver limits.

## 2. PagedAttention (Legacy Python)
For LLM inference in the Python version, VNN uses a `BlockTable` to map token sequences to scattered physical blocks in VRAM. This eliminates fragmentation but incurs Python GIL overhead during virtualization.
- **File**: `paged_attention.py`
- **Mechanism**: Dynamic mapping of non-contiguous fragments token-by-token.

## 3. Kaggle Remote Compute (Legacy Exclusive)
One of the most unique features of the legacy library is the ability to offload the heaviest computations to the cloud.
- **Activation**: Triggers when operation size > `VNN_KAGGLE_THRESHOLD`.
- **Workflow**: 
  1. Local VNN tiles the operation.
  2. Data is uploaded to Kaggle via public API.
  3. A remote Kaggle Notebook executes PyTorch/CUDA kernels.
  4. Result is downloaded back to the local SSD cache.

## 4. History and Context
This architecture was the foundation of VNN. While stable, it is limited by Python's execution speed. It serves as the functional blueprint for the ongoing Rust port (VNN Rusted).
