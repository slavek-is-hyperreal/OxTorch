# MSTS Logic: The Unified Dispatcher (v3.8.1-rc)

MSTS (Mera Style Tiling System) is the central orchestration layer of OxTorch. In the **Unified Architecture**, MSTS is the **Sole Decision Maker** for all CPU operations, balancing RAM cache hits and SSD streaming bandwidth.

---

## 1. The Multi-Path Dispatch (Automatic Selection)

When a binary operation (e.g., `add`) is called, `dispatch_binary_op` (in `msts.rs`) analyzes the tensors and chooses a path based on memory location and hardware thresholds:

### Path A: RAM-FastPath (All in RAM)
- **Threshold**: `is_any_ssd == false`.
- **Mechanism**: **Optimized SIMD dispatch**.
- **Reason**: Tiling overhead is skipped to minimize latency for tensors that already fit in RAM.
- **Implementation**: Directly calls `core_ops` (SIMD) on the raw memory slices.

### Path B: Hybrid / SSD (Tiled MSTS v2)
- **Mechanism**: **execute_op_unified + CrookScheduler**.
- **Reason**: Decouples disk I/O from computation. 
- **Sub-Paths**:
    1.  **SSD Direct**: If the tensor is very small (< `DIRECT_MAX`), it loads the whole chunk into an `AlignedBuffer` and computes serially.
    2.  **Tiled Ring**: If the tensor is large, it uses a triple-buffered ring (`CrookScheduler`).
        - `TILE_SMALL`: Cache-Native (75% L2) — matched for peak compute performance in RAM.
        - `TILE_LARGE`: IO-Native (8MB) — matched for maximum SSD/NVMe bandwidth (O_DIRECT).

---

## 2. Multi-Stream Bitmask Barrier (MERA-400 Handshake)

For binary operations (e.g., `A + B = C`), the `CrookScheduler` uses a **Bitmask Barrier** (`ready_bits`) instead of a simple linear state machine. This allows asynchronous readers for `A` and `B` to work in parallel.

1.  **Requirement**: Binary ops expect `ready_bits == 0b11` (Decimal 3).
2.  **Reader A**: Fetches tile `N` for Source A. Sets `Bit 0` (OR 1).
3.  **Reader B**: Fetches tile `N` for Source B. Sets `Bit 1` (OR 2).
4.  **Compute Barrier**: The compute loop spins on `ready_bits.load() == 3`. 
5.  **Compute**: Transitions to `TILE_COMPUTING` and executes the Leaf Kernel.
6.  **Write Marker**: Transitions to `TILE_READY_FOR_WRITE`.

---

## 3. Hardware Discovery (build-time)

The thresholds and tile sizes are burned into the binary at compile time via `build.rs`:

- **Threshold Discovery**: Queries indices in `/sys/devices/system/cpu/cpu0/cache/` to detect L2 and L3 sizes.
- **Strategy Constants**:
    - `DIRECT_MAX`: Set to 50% of L3 cache size.
    - `TILE_SMALL`: Set to 75% of L2 cache size.
    - `RING_LARGE`: Ring depth bounded by total L3 size (to prevent cache thrashing).
    - `CAPACITOR_FLOOR_MB`: 25% of build-time system RAM.

---

## 4. Resource Policy: Zero OOM

- **Deterministic Memory**: The memory footprint is fixed by the `ring_size` and `tile_size`. 
- **Example**: Processing a 100GB tensor with 8 tiles of 8MB each uses only ~192MB of "working RAM" (3 slots per tile: A, B, Result). 
- **Scalability**: This allows the largest LLMs to run on standard desktops with consistent, predictable performance.
