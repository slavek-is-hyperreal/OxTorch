# OxTorch SSD Storage Guide (v3.8.1-rc)

OxTorch supports Multi-Source Tensor Streaming (MSTS), allowing you to process datasets that are much larger than your available RAM. Large tensors can be mapped directly to files on NVMe/SSD via `io_uring` and `O_DIRECT`.

---

## 1. Binary Storage Format

To ensure maximum performance (via zero-copy DMA), OxTorch uses a **strictly raw** binary format:

1.  **No Headers**: The file must contain only the raw numerical data. No metadata, JSON, or magic numbers.
2.  **C-order (Row-Major)**: Data must be stored in contiguous row-major order.
3.  **Alignment**: **IMPORTANT:** Files must be stored on systems with at least 512-byte block alignment. Internally, OxTorch uses **4096-byte aligned** memory buffers for all SSD operations to ensure compatibility with NVMe `O_DIRECT`.
4.  **Data Types**:
    - `float32` (4-byte), `float16` (2-byte), `bf16` (2-byte).
    - `int8` (1-byte).
    - `BitNet` / `I2_S` (Sub-byte quantized packed formats).

---

## 2. Dispatch Mechanics: SSD Direct vs. MSTS

MSTS automatically selects how to read and compute SSD data based on the tensor's size:

### SSD Direct Path
- **Threshold**: Tensor size <= `DIRECT_MAX` (typically ~3MB-8MB, 50% of L3 cache).
- **Behavior**: OxTorch loads the *entire* tensor into a single aligned buffer and executes a serial kernel.
- **Reason**: The overhead of setting up a triple-buffered ring is not worth it for small results.

### Tiled Hybrid Path
- **Threshold**: Tensor size > `DIRECT_MAX`.
- **Behavior**: Uses the **CrookScheduler** to stream tiles in 8MB chunks.
- **Overlap**: Compute on Tile N occurs while Tile N+1 is being read from disk and Tile N-1 is being written. This hides disk latency completely.

---

## 3. High-Speed Prefetching

For sequential models, use `tensor.prefetch_ssd()`. This spawns a background Parallel-Async-Reader (PAR) that proactively fills the **Global RAM Capacitor** with up to 512MB of in-flight data, ensuring the compute engine never waits for the ring buffer to fill.

---

## 4. O_DIRECT Caution
SSD-backed tensors use `O_DIRECT`. This means the file size must **exactly match** the calculated tensor bytes (`shape * dtype_size`). If the file is smaller or truncated, OxTorch will throw an I/O error during the MSTS handshake.
