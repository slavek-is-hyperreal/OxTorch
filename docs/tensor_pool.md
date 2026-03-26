# TensorPool: Hot-Path Slab Allocator

`TensorPool` is a high-performance, **Thread-Local Slab Allocator** designed to eliminate system allocations (`malloc`/`free`) in OxTorch's hot loops.

## 1. Why TensorPool?
Many deep learning operations (e.g., LayerNorm in F16) require temporary buffers for intermediate data (e.g., conversion to F32). Performing `Vec::with_capacity` allocations for every row of a tensor (row-wise) incurs massive overhead and memory fragmentation. `TensorPool` solves this by recycling buffers.

## 2. Architecture
`TensorPool` is implemented as a `ThreadLocal` structure, meaning each worker thread (e.g., in Rayon) has its own pool and does not compete for locks (lock-free access).

### Bucketing System
The pool manages 6 pre-allocated buckets of varying sizes:
- **Tiny**: < 4 KB
- **Small**: < 64 KB
- **Medium**: < 1 MB
- **Large**: < 16 MB
- **X-Large**: < 256 MB
- **Massive**: > 256 MB

## 3. Usage in Code (Rust)

To retrieve a temporary buffer from the pool:

```rust
use crate::tensor::pool::TensorPool;

fn my_kernel(data: &[f16]) {
    // Get an f32 buffer of the required size
    let mut workspace = TensorPool::get_f32_buffer(data.len());
    
    // Perform computations...
    for i in 0..data.len() { workspace[i] = data[i].to_f32(); }
    
    // The buffer is automatically RETURNED to the pool when 'workspace' finishes (Drop trait)
}
```

## 4. Safety Rules
1. **Zero-Copy**: `TensorPool` returns a `&mut [T]` or a smart pointer, ensuring that data is not copied when being retrieved from the pool.
2. **Alignment**: All buffers are aligned to 64-byte boundaries, as required for AVX-512 instructions.
3. **Hot-Swap**: If the required size exceeds the current bucket capacity, `TensorPool` performs a one-time system allocation and keeps it in the pool for future use.

## 5. MSTS Integration
When operating in **SSD Streaming (MSTS Path C)** mode, data tiles are loaded directly into buffers sourced from `TensorPool`. This allows a 16GB model streaming process to run with a constant RAM footprint in the range of a few hundred megabytes.
