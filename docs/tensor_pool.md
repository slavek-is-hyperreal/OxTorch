# TensorPool: The MSTS Resource Model (v3.8.1-rc)

In the **v3.8.1-rc** architecture, the dynamic, bucketed slab allocator has been replaced by a more deterministic, **Pre-allocated Ring Buffer** system managed by the `CrookScheduler`.

---

## 1. Role in the Pipeline

Instead of a thread-local pool of various sizes, OxTorch now focuses on **Resource Determinism**. When an operation is dispatched via MSTS:

1.  **MSTS**: Defines the optimal `ring_size` (e.g., 8-16) and `tile_size` (e.g., 8MB) based on hardware.
2.  **CrookScheduler**: Allocates a fixed vector of **StatefulTiles**.
3.  **Tile Persistence**: These tiles are recycled continuously for the duration of the tensor operation. Memory is never freed or re-allocated during a large SSD stream.
4.  **Zero-Allocation**: This ensures a perfectly flat memory footprint, preventing OOM when processing datasets much larger than system RAM.

---

## 2. Global RAM vs. Local Ring

- **Capacitor (Global)**: A single, massive reservoir (up to 50% RAM) that caches SSD raw blocks.
- **Crook Ring (Local)**: A set of high-speed "Working Bins" (Tiles) that are active during execution.
- **Transporter**: `io_uring` fills the Ring from either the Global Capacitor (Zero-Copy) or directly from the NVMe controller.

---

## 3. Guarantees: Alignment & Direct I/O

- **4096-byte Alignment**: Every slot in a `StatefulTile` is strictly aligned to **4096 bytes** using `std::alloc::alloc`.
- **Reason**: This is a hard requirement for **O_DIRECT** on modern NVMe and ZFS filesystems. It allows the kernel to bypass the page cache and write DMA directly from the device to our tile buffers.
- **SIMD Optimized**: 4096-byte alignment naturally fulfills the 64-byte alignment requirement for AVX2 and NEON SIMD kernels.

---

## 4. Implementation Details

The `TileSlot` struct (in `crook_scheduler.rs`) is the core unit of the pool:

```rust
pub struct TileSlot {
    pub local_buf: UnsafeCell<AlignedBuffer>,
    pub capacitor_ptr: AtomicPtr<u8>,
    pub size: usize,
}
```

- **Local Buffer**: Used for standard streaming reads.
- **Capacitor Pointer**: When data is already in the Global RAM Capacitor, this pointer is set to the capacitor's memory, achieving **Zero-Copy Hybrid Computing**.
