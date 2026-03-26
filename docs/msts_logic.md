# MSTS: Mera Style Tiling System

MSTS is the heart of the OxTorch architecture, inspired by the oscillatory and asynchronous nature of the MERA-400 minicomputer. This system allows for the processing of tensors that exceed available RAM (Out-of-Core) by streaming data from the SSD directly to the computing units (CPU/GPU).

---

## 1. Native MSTS vs. MSTS PyTorch (Fallback)

> [!IMPORTANT]
> Understanding the difference between these two mechanisms is critical for system performance.

| Feature | Native MSTS (OxTorch) | MSTS PyTorch (Fallback) |
|:---|:---|:---|
| **Language** | Pure Rust (`src/tensor/msts.rs`) | Rust + Python (`oxtorch/tensor.py`) |
| **Kernels** | Custom SIMD (AVX/NEON) | Any `torch.*` function |
| **Performance** | Maximum (zero GIL, zero allocations) | Lower (GIL overhead, NumPy conversion) |
| **Purpose** | Production operations (ReLU, MatMul) | Rapid support for new ops / prototypes |
| **DType Support** | Full (F32, F16, BF16, I8, Ternary) | Dependent on NumPy mapping |

### How does the Fallback work?
The `msts_pytorch_apply` mechanism in Python intercepts calls for operations that do not yet have a native SSD counterpart. It slices the SSD tensor into tiles (usually 1MB), converts each tile to NumPy, sends it to PyTorch, receives the result, and writes it back to the SSD. This allows using `torch.erf()` or `torch.exp()` on 100GB tensors without running out of memory (OOM).

---

## 2. StatefulTile State Machine

Each "tile" in the circular buffer (`CrookScheduler`) has an atomic state tag (`AtomicU32`) that controls I/O orchestration:

1.  **`TILE_EMPTY` (0)**: The tile is empty. The Reader thread (PPU) can begin loading data from the SSD.
2.  **`TILE_READING_FROM_DISK` (1)**: `io_uring` is filling the buffer.
3.  **`TILE_READY_FOR_COMPUTE` (2)**: Data is in RAM. The compute unit (CPU/GPU) can claim the tile.
4.  **`TILE_COMPUTING` (3)**: The kernel is performing operations on the data. Other threads cannot "steal" this tile.
5.  **`TILE_READY_FOR_WRITE` (4)**: The computation result is ready to be flushed to disk.
6.  **`TILE_WRITING_TO_DISK` (5)**: `io_uring` is writing data to the `.ssd` file. After completion, the tile returns to the `EMPTY` state.

---

## 3. 3-Path Dispatch (Automatic Path Selection)

OxTorch does not use a single strategy for all sizes. The performance threshold is "burned-into" the binary during compilation (`build.rs`) based on the L2/L3 cache parameters of the target machine.

### Path A: Direct (Tensor < `DIRECT_MAX` ≈ 3MB)
*   **Mechanism**: Direct `read_chunk` into `AlignedBuffer` and write.
*   **Advantage**: Zero overhead for thread creation, atomic synchronization, or ring-buffer orchestration. Ideal for small parameter vectors.

### Path B: Single-thread (Tensor < 32MB)
*   **Mechanism**: 1 I/O thread (read-only), computation performed inline on the main thread.
*   **Ring Size**: `RING_SMALL` (2 tiles).
*   **Tile Size**: `TILE_SMALL` (Targeting 75% L2 cache to keep data "hot").

### Path C: Full CrookScheduler (Tensor >= 32MB)
*   **Mechanism**: Full orchestration. 2 dedicated I/O threads (Background Read & Write) + Rayon parallel compute.
*   **Ring Size**: `RING_LARGE` (Dynamically selected based on L3, max 8).
*   **Tile Size**: `TILE_LARGE` (Usually 2-4MB, optimal for sequential SSD burst).

---

## 4. CrookScheduler Orchestration

The scheduler implements an asynchronous "Race for Tiles" model. In `Path C`:
- The **Reader** thread races ahead, filling `EMPTY` tiles from the SSD.
- The **Compute** thread (main/Rayon) waits for `READY_FOR_COMPUTE`, processes, and hands it off.
- The **Writer** thread follows the Compute thread, flushing `READY_FOR_WRITE` to the SSD.

This eliminates stalls (I/O wait) — while the CPU computes `Tile 5`, the SSD is loading `Tile 7` and writing `Tile 3`.
