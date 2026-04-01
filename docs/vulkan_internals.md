# Vulkan Backend Internals

OxTorch's Vulkan backend is a high-performance compute engine built directly on the raw **Ash** (Vulkan 1.2) bindings. It avoids heavy abstractions to minimize overhead and maximize control over GPU resources.

## Core Architecture

The backend is centered around the `AshBackend` singleton (initialized via `init_backend`). It manages the Vulkan instance, physical/logical devices, and compute/transfer queues.

### Key Technologies
- **Ash**: Zero-overhead Rust bindings for Vulkan.
- **gpu_allocator**: Handles efficient VRAM allocation and tracking.
- **Timeline Semaphores**: Used for fine-grained, asynchronous synchronization between the Host (CPU) and Device (GPU).

## Resource Management

### VRAM Pooling & Caching
To avoid the high cost of frequent `vkAllocateMemory` calls, OxTorch uses a two-tier strategy:
1.  **Buffer Cache**: A global `buffer_cache` (`Vec<CachedBuffer>`) stores recently freed buffers for immediate reuse.
2.  **VRAM Slab Allocator**: A large (256MB) `pool_buffer` is pre-allocated at startup. Small `GpuOnly` buffers are sub-allocated from this slab using a free-list manager (`pool_free_list`).

### Descriptor Management
To support asynchronous execution without race conditions, the backend uses **Rotating Descriptor Set Pools**. 
- Each operation type (e.g., `elementwise`, `matmul`) has its own `DescriptorSetPool`.
- The pool contains a fixed number of pre-allocated `vk::DescriptorSet`.
- Each call to `execute_*` fetches the `next()` descriptor set and updates it via `vkUpdateDescriptorSets` to point to the current input/output buffers.

## Execution Pipeline

Every Vulkan operation follows a standardized execution flow to ensure data consistency and performance:

1.  **Host-to-Staging Upload**: Data is copied from CPU RAM into a `CpuToGpu` staging buffer. If the data is F16 or BF16, it is converted to F32 during this step.
2.  **Staging-to-Device Copy**: A `vkCmdCopyBuffer` command is recorded to move data into the high-speed `GpuOnly` memory.
3.  **Compute Dispatch**: 
    - Descriptor sets are bound.
    - Op-specific metadata is passed via **Push Constants** (expanded to **44 bytes** in v8.2 to support 2D Strides).
    - The compute shader is dispatched with `vkCmdDispatch`. For Tiled MatMul, this uses a 16x16 local workgroup size.
4.  **Device-to-Staging Copy**: Results are moved from `GpuOnly` memory back to a `GpuToCpu` readback staging buffer.
5.  **Synchronization**: 
    - `vkCmdPipelineBarrier` ensures that Transfers finish before Compute starts, and vice-versa.
    - A **Timeline Semaphore** value is incremented and signaled upon queue completion.
6.  **Readback & Cleanup**:
    - The Host waits for the target timeline value.
    - The `CachedBuffer` objects are recycled.
    - Results are downloaded from the staging buffer (converting back from F32 if necessary).

## Shader Structure
Shaders are written in GLSL and compiled to SPIR-V via `naga`. 
- **MSTS v2 Layout**: Incorporates **2D Stride-Aware indexing** (`idx = row * s_row + col * s_col`).
- **Tiled Execution**: Uses Local Data Share (LDS) to cache matrix tiles, significantly reducing VRAM bandwidth pressure for MatMul operations.
- **Push Constants**: Used for dynamic parameters: M, N, K, 6 strides (A, B, C), and 2 offsets.

## Asynchronous Operations
Asynchronous tasks are tracked via the `AsyncOp` struct. The `poll_async_ops` function is called periodically to check for completed tasks and reclaim their associated staging and device buffers.
