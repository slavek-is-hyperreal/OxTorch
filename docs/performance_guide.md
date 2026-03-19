# Performance Guide (v3.7.0 (The BitNet Leapfrog))

This guide explains how to interpret benchmark results and how to maximize throughput
on the target hardware (i5-3450, AMD Bonaire R7 260X, 24GB DDR3, ZFS SSD pool).

---

## 1. Reading Benchmark Output

The `unified_benchmark.py` harness reports:

- **PT (Median)**: PyTorch execution time, median over N runs. The primary reference.
- **OxTorch (Median)**: OxTorch execution time, median over N runs.
- **OxTorch (Std)**: Standard deviation of OxTorch time. High### v3.7.0 (The BitNet Leapfrog))

| Operation | DType | Mode | Ratio (OxTorch / PT) | Why? |
| :--- | :--- | :--- | :--- | :--- |
| MatMul | F32 | CPU | **0.82x** | Optimized BLAS integration |
| Softmax | F32 | CPU | **0.57x** | Masked vectorized EXP |
| ReLU | F16 | CPU | **0.66x** | Native F16C / AVX1 path |
| MatMul | F16 | CPU | **0.002x** | PT lacks CPU F16C optimization |
| GELU | F32 | CPU | 11.2x | **Under Optimization (Phase 7)** |
| ReLU | SSD | Hybrid | N/A (PT: OOM) | MSTS / io_uring threading |
PyTorch falls back to scalar software emulation without AVX-512 FP16).
- **Parity**: Pass/fail of `numpy.testing.assert_allclose`. A pass means numerical
  output matches PyTorch within the precision tolerance of the dtype.

---

## 2. Hardware Characteristics and Expected Results

### CPU (i5-3450, Ivy Bridge)

- 4 cores, no hyperthreading, 6MB L3 cache
- Has AVX and F16C but NOT AVX2 or FMA
- F16/BF16 operations dispatch to F16C intrinsics (F16) or SSE2 SWAR (BF16)
- F32 MatMul: near 1:1 with PyTorch (~0.21s for 2048x2048)
- F16/BF16 MatMul: ~500x faster than PyTorch, which uses scalar emulation

### GPU (AMD Radeon R7 260X, Bonaire GCN 1.1)

- 1GB GDDR5 VRAM
- PCIe 3.0 x16 bus, but PCIe staging roundtrip overhead is ~80ms on this card
- No native FP16 compute: all shader math runs in F32, with F16/BF16 stored in system RAM
- Break-even point for dispatching to GPU: approximately 4M elements (~16MB F32)
- Below this threshold, Vulkan PCIe staging cost dominates and CPU SWAR is faster
- Vulkan ReLU on 1M elements: ~85ms (26x slower than PyTorch)
- Vulkan ReLU on 15M+ elements: expected to approach CPU performance or beat it

### SSD (ZFS pool)

- Effective sequential read throughput: ~80-90 MB/s for the 16GB Monster ReLU benchmark
- io_uring O_DIRECT at 1MB ZFS recordsize boundaries eliminates all page cache overhead
- ZFS recordsize must be set to 1MB for optimal alignment: `zfs set recordsize=1M pool/dataset`

---

## 3. Thermal Behavior

Observed across long benchmark sessions (9000+ seconds):

- **Cold start**: L3 cache is empty, first runs take slightly longer.
- **Warm**: After a few minutes, tile sizes are cache-resident and times stabilize.
- **Heat soak (after 2+ hours)**: CPU and GPU clock-down by 10-15%. The OxTorch/PT ratio
  remains stable because both are equally affected. Use Median to filter isolated spikes.

StdDev thresholds:
- Below 5ms: excellent stability
- 5-50ms: normal for background OS activity
- Above 50ms: likely thermal throttling or memory pressure from other processes

---

## 4. GPU Dispatch Threshold

`VULKAN_MIN_ELEMS = 4_194_304` (4M elements, ~16MB F32, ~8MB F16)

This constant in `src/tensor/mod.rs` controls the hybrid tile-pulling dispatcher:
- Below the threshold: only CPU SWAR workers run. GPU dispatcher thread is not spawned.
- At or above the threshold: GPU dispatcher competes for tiles alongside CPU workers.

To tune this for different hardware, search for `VULKAN_MIN_ELEMS` in `src/tensor/mod.rs`.
On GPUs with lower PCIe latency (e.g., discrete desktop cards with fast transfers),
the threshold can be lowered significantly.

---

## 5. Maximizing Throughput

1. **Use `device="cpu"` for small tensors** (below ~2M elements): the Vulkan overhead is
   not amortized on the Bonaire. CPU via Rayon + SIMD is faster.

2. **Use `device="vulkan"` for F16/BF16 MatMul**: even though the GPU cannot natively
   compute F16, the SWAR upcast in Rust is dramatically faster than PyTorch's scalar path.

3. **Use `device="hybrid"` for large activations** (>4M elements): tile-pulling lets the
   GPU handle some of the work while CPU SWAR handles the rest, in parallel.

4. **Use `Tensor.from_ssd` for out-of-core weights**: this is the io_uring path. Never
   load large tensors with numpy and pass them to the Tensor constructor if they do not
   fit comfortably in RAM below the `l2_ram_max_bytes` budget threshold.

5. **Reduce background processes** during long benchmark runs to minimize StdDev noise.

---

## 6. Known Limitations

- **Small GPU dispatches are expensive on Bonaire**: The ~80ms PCIe staging cost means
  Vulkan is a net negative below 4M elements. This is a hardware characteristic, not a bug.
- **F16 MatMul Hybrid**: the tile-pulling dispatch (Phase 4) currently covers only
  activation functions. MatMul hybrid still uses a full-tensor Vulkan dispatch.
- **No native FP16 shader compute**: Bonaire GCN 1.1 does not support FP16 in compute
  shaders via Vulkan. All SPIR-V shaders operate on FP32. F16/BF16 data is converted
  on the CPU before upload and after download.
