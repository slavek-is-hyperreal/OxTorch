# OxTorch Execution Modes

OxTorch supports four main execution strategies that allow the engine to adapt to data size and available hardware infrastructure.

---

## 1. CPU Mode (`device="cpu"`)

This mode is oriented towards low latency and maximum utilization of the processor's SIMD instructions.

*   **SIMD-only Isolation**: Kernels in OxTorch are written as single-threaded functions. We avoid using `rayon` or `std::thread` inside a single operation (e.g., `add_f32`) to prevent massive data exchange in L2/L3 cache (cache thrashing).
*   **Prefetching**: We utilize `_mm_prefetch` (x86) or `PRFM` (ARM) to load subsequent cache lines before their actual use in the SIMD loop.
*   **When to use?**:
    - Small and medium tensors (< 4M elements).
    - Operations requiring frequent random access (e.g., complex reductions).
    - When the GPU is busy with other tasks.

---

## 2. Vulkan GPU Mode (`device="vulkan"`)

A high-throughput mode utilizing the computing power of the graphics card via the Vulkan 1.2 API (using the `ash` library).

*   **Raw Ash Setup**: We do not use high-level wrappers. We directly manage `CommandPool`, `PipelineLayout`, and `DescriptorSet`.
*   **Timeline Semaphores**: For asynchronous synchronization, we use timeline semaphores, allowing GPU progress tracking without blocking the main thread (zero-block polling).
*   **PCIe Bottleneck (Legacy Hardware)**: On older hardware (PCIe 3.0), transfer costs to `staging` buffers are approximately 80ms.
*   **Break-even Point**: The GPU only becomes faster than the CPU above **4,194,304 elements** (4M). Below this threshold, the dispatch and transfer overhead exceeds the computational gain.

---

## 3. Hybrid Mode (`device="hybrid"`)

A unique OxTorch mode inspired by the MERA-400 architecture. It implements a "Race for Tiles" strategy.

*   **Mechanism**: The operation is divided into Tiles. The GPU thread and CPU threads (Rayon) use a shared atomic counter `AtomicUsize`.
*   **Race for Tiles**: Each hardware unit (GPU, Core 0, Core 1...) "claims" the next available tile from the queue, processes it, and requests the next one.
*   **Advantage**: The system automatically balances the load. If the GPU is slow due to PCIe transfer, the CPU takes over more tiles. If the CPU is busy with I/O, the GPU catches up. It is designed for full hardware saturation (100% CPU and GPU load).

---

## 4. SSD Streaming Mode (`device="ssd"`)

The "infinite memory" mode, allowing 70B+ models to run on machines with only 8GB of RAM.

*   **io_uring + O_DIRECT**: We use the latest Linux API for asynchronous I/O. By using the `O_DIRECT` flag, we bypass the system Page Cache, eliminating double buffering and unpredictable RAM usage spikes.
*   **Aligned 1MB Records**: Data on the SSD is aligned to 1MB, which maps perfectly to `recordsize=1M` on ZFS filesystems. This allows for DMA (Direct Memory Access) transfers directly from the disk controller to the CPU buffers.
*   **CrookScheduler**: A ring buffer (Ring Buffer) manages data flow. While the CPU computes `Tile N`, the background system loads `Tile N+1` from the SSD and writes `Tile N-1`.
