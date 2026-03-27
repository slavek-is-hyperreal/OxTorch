# TensorPool: The Hot-Path Memory Provider (NEW)

`TensorPool` is a thread-local slab allocator. In the **Unified Architecture**, its primary role is to provide the "Tiles" (Working Bins) used by the MSTS Orchestrator.

---

## 1. Role in the Pipeline

When the **MSTS Orchestrator** needs to process a 10GB tensor, it doesn't allocate memory for each 4MB chunk. Instead, it asks **TensorPool** for a "BufferGuard".

1.  **MSTS**: "Give me 8 tiles of 4MB each."
2.  **TensorPool**: Returns 8 pre-allocated `Vec<u8>` wrappers (BufferGuards).
3.  **MSTS**: Uses these tiles in the `CrookScheduler` ring.
4.  **Leaf Kernels**: Process the data inside the tiles.
5.  **MSTS**: Once a tile is written back to SSD, the **BufferGuard** is dropped.
6.  **TensorPool**: The `Drop` implementation automatically returns the buffer to the pool (lock-free).

---

## 2. Bucketing Strategy

TensorPool manages 6 buckets, each targeting a specific memory class. In the new architecture, **Bucket 3 (Large)** and **Bucket 4 (X-Large)** are the "Workhorses" for MSTS.

- **Bucket 0 (Tiny)**: < 4 KB
- **Bucket 1 (Small)**: < 64 KB
- **Bucket 2 (Medium)**: < 1 MB
- **Bucket 3 (Large)**: < 4 MB (Standard MSTS Tile)
- **Bucket 4 (X-Large)**: < 16 MB (Enhanced MSTS Tile for large caches)
- **Bucket 5 (Massive)**: < 64 MB (Special usage)

---

## 3. Guarantees: Alignment & Zero-Allocation

- **64-byte Alignment**: All buffers in the pool are guaranteed to be 64-byte aligned (aligned to a cache line and SIMD boundary). This is critical for **AVX-512** and **NEON** kernels.
- **Zero-Allocation Hot-Loop**: Once the pool is "warmed up", NO system calls (`malloc`/`mmap`) are made during inference. This results in deterministic, low-jitter performance.
- **Local vs Global**: Unlike the **Capacitor** (which is a single global reservoir), **TensorPool** is thread-local. This avoids mutex contention during high-speed tile recycling.

---

## 4. Integration Example

The new `msts.rs` logic uses `TensorPool` like this:

```rust
// Inside a worker thread:
let mut tile_buffer = TENSOR_POOL.with(|p| p.borrow_mut().alloc(tile_size));
// ... fill from Capacitor via io_uring ...
// ... run Leaf Kernel ...
// ... result automatically returns to pool on function exit ...
```

---

## 5. Memory Safety

- **BufferGuards**: These RAII wrappers ensure that a buffer is **NEVER** leaked and is always returned to the pool, even if a kernel panics.
- **Poisoning**: Each time a buffer is returned to the pool, we perform a zero-fill if security is a concern (though for inference, we usually prioritize speed).
