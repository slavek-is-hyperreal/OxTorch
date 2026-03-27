# MSTS Logic: The Unified Dispatcher (NEW)

MSTS (Mera Style Tiling System) is the central orchestration layer of OxTorch. In the **Unified Architecture**, MSTS is no longer just for SSD; it is the **Sole Decision Maker** for all CPU operations.

---

## 1. The 3-Path Dispatch (Automatic Selection)

When an operation (e.g., `add`) is called, MSTS analyzes the tensors and selects the most efficient execution path:

### Path A: Direct (Small & RAM)
- **Threshold**: `Tensor < L3_CACHE_SIZE / 2`.
- **Mechanism**: **Single-threaded, Serial Call**.
- **Reason**: The overhead of spawning a thread pool (Rayon) or managing an I/O ring is greater than the compute time for small data.
- **Data**: Stays in place. No extra buffer allocation.

### Path B: Parallel (Large & RAM)
- **Threshold**: `Tensor >= L3_CACHE_SIZE / 2`.
- **Mechanism**: **Rayon `par_chunks_mut`**.
- **Reason**: RAM is fast enough that I/O isn't the bottleneck. We parallelize the compute across all cores.
- **Tiling**: MSTS splits the RAM tensor into tiles that fit in **L2 cache**.

### Path C: Streaming (SSD Involved)
- **Mechanism**: **CrookScheduler (Full Orchestration)**.
- **Reason**: SSD latency is the bottleneck. We must decouple I/O from Compute.
- **Process**:
    1. **PPU (Reader)**: Fills the **Capacitor** from SSD as fast as possible.
    2. **MSTS**: Requests tiles from **TensorPool**.
    3. **PPU (Transporter)**: `io_uring` copies from Capacitor into TensorPool tiles.
    4. **CPU (Compute)**: Runs the **Leaf Kernel** (Serial) on ready tiles.
    5. **PPU (Writer)**: Flushes the result to SSD.

---

## 2. StatefulTile Binary Flow

For binary operations (e.g., `A + B = C`), the `CrookScheduler` manages a **State Machine** for each "token" (tile):

1.  **`TILE_EMPTY`**: Ready to receive data.
2.  **`TILE_READING_A` / `TILE_READING_B`**: `io_uring` is streaming the two operands from the Capacitor (or disk) into the tile's buffers.
3.  **`TILE_READY_FOR_COMPUTE`**: Both operands are in RAM.
4.  **`TILE_COMPUTING`**: The **Leaf Kernel** is adding them. 
5.  **`TILE_READY_FOR_WRITE`**: Result is ready.
6.  **`TILE_WRITING`**: `io_uring` is flushing to the output SSD file.

---

## 3. Hardware Alignment (build.rs)

The thresholds and tile sizes are not hardcoded. They are determined during the `cargo build` phase by `build.rs`, which queries the host's CPU topology (L1/L2/L3 cache sizes). 

- **L1 Alignment**: Small rows fit here.
- **L2 Alignment**: Leaf Kernels target this for maximum speed.
- **L3 Alignment**: MSTS uses this to decide when to switch from Path A to Path B.

---

## 4. Resource Policy: Zero Warnings, Zero OOM

- **Capacitor**: Fixed at 50% RAM.
- **TensorPool**: Fixed number of pre-allocated tiles (e.g., 8-16).
- **Control**: By limiting the number of active tiles, OxTorch can process a 1TB model on a 16GB machine with a deterministic memory footprint.
