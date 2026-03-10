# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.4.0] - 2026-03-10 "Iron Age Complete"

### Added
- **Official Raw Vulkan Backend**: The experimental `ash` rewrite from `dev_raw_vulkan` has been merged into the main `dev` branch. `wgpu` is officially retired in this release.
- **VRAM Memory Safety**: Added a strict 512MB VRAM cache budget, aggressive PRUNE capability, and "Retry-on-OOM" loop in `backend.rs` to prevent system hangs on low-VRAM GPUs.
- **Python-Side GC**: Added explicit `gc.collect()` and `del` in benchmark suites for processing massive 15M-element tensors within 24GB RAM limits.

### Changed
- All project versions bumped from `v3.3.0` to `v3.4.0` to mark the stabilization of the "Iron Age" capabilities.
- `vulkannn_rusted_dev` python module now natively uses the raw Vulkan bindings.

---

## [3.3.0] - 2026-03-10 "Iron Age" (Experimental)

### Added
- **MSTS Tile-Pulling Hybrid Dispatch (Phase 4)**: Replaced the static 30/70 CPU/GPU split with a
  dynamic atomic tile counter (`Arc<AtomicUsize>`). One GPU dispatcher thread and one CPU SWAR
  thread race to claim 256K-element tiles. No locks, no static allocation. The faster resource
  naturally claims more work, embodying the MERA-400 CROOK OS tagged-token dataflow principle.
- **`execute_activation_chunked`**: New `backend.rs` API that processes a sub-range of elements
  (by offset + count) to support the tile-pulling hybrid dispatch.
- **GPU Dispatch Threshold (`VULKAN_MIN_ELEMS = 4M`)**: For tensors below 4M elements (~16MB F32),
  Vulkan PCIe staging overhead (~80ms on Bonaire) exceeds compute time. The GPU dispatcher is
  skipped entirely below this threshold; all tiles are claimed by CPU SWAR workers.
- **Cross-Platform SIMD Fallback Chain (`avx_swar.rs`)**: Complete rewrite with runtime dispatch
  covering: F16C+AVX (Ivy Bridge, Haswell) -> SSE2 SWAR branchless (any x86_64, no F16C needed)
  -> AArch64 NEON -> scalar Rayon. All four conversion directions (F32/F16/BF16 in both ways).
  Note: the i5-3450 (Ivy Bridge) DOES have F16C -- it dispatches to hardware intrinsics.
- **Branch-Specific Module Naming**: Each branch compiles to a distinct Python module name for
  parallel A/B benchmarking (`vulkannn_rusted_main`, `_test`, `_dev`, `_exp`).
- **Branch-Aware Benchmark Import**: `unified_benchmark.py` now dynamically imports the active
  branch module via a fallback chain. No manual edits needed when switching branches.
- **Raw `ash` Vulkan Backend** (merged from dev_raw_vulkan): Complete rewrite from `wgpu`/WGSL to
  `ash` (raw Vulkan 1.2). Explicit compute and transfer command pools, Timeline Semaphores for async
  GPU operation chaining, buffer recycling cache.
- **Out-of-Core `io_uring` Engine**: `src/io_uring_engine.rs` streams data with `O_DIRECT` at
  1MB ZFS recordsize boundaries, bypassing the Linux VFS page cache.
- **MERA Style Task Scheduler (MSTS)**: `src/crook_scheduler.rs` implements `StatefulTile`, a
  lockless ring buffer with atomic state transitions inspired by the MERA-400 CROOK OS.
- **15M Element ReLU Benchmark**: Added to `unified_benchmark.py` across all dtypes and modes to
  reveal the GPU dispatch break-even point above the 4M element threshold.

### Fixed
- **Incorrect F16C claim**: The research document incorrectly stated the i5-3450 lacks F16C.
  It does have F16C. The SWAR path is retained as a fallback for genuinely F16C-less CPUs.
- **Removed dead SSSE3 mask code**: Cleaned up the unreachable SSSE3 branch in the SSE2 BF16
  conversion path.

### Changed
- `unary_op` in `tensor.rs`: the `hybrid` device path now calls the tile-pulling ring instead of
  the single whole-tensor `execute_activation` call.
- Version bump to reflect the scope of the ash rewrite and Phase 4 hybrid engine.

---

## [3.2.0] - 2026-03-09 "Valkyrie"

### Added
- **Tri-Precision Engine**: Native F32, F16, BF16 support with CPU fast paths.
- **Statistical benchmark harness**: `--runs N` with Median, Mean, StdDev, and history logging.
- **Session duration tracking**: `total_duration_seconds` recorded per run for thermal analysis.
- **Documentation overhaul**: README, api_reference, architecture, performance_guide rewritten.

### Fixed
- PyTorch F16/BF16 parity tolerance tuning.
- Median used as primary metric to resist OS context-switch spikes.

---

## [2.9.0] - 2026-03-04

### Added
- Coefficient of Variation (CV%) and P95 percentile tracking in benchmark harness.
- API Reference documentation with source line references.
- Hardware-invariant Ratio (VNN/PT) as primary regression metric.

---

## [2.8.0] - 2026-03-03

### Added
- CPU near-parity with PyTorch for large RAM-resident MatMul/ReLU via Rayon + matrixmultiply.
- Unified benchmark (`unified_benchmark.py`) for continuous parity and performance monitoring.
- WGSL compute shaders with 256-thread workgroups.

### Changed
- Repository restructured: original Python/Taichi library archived to `Python_Legacy/`.
- All Rust compiler warnings eliminated.

---

## [2.5.0] - 2026-03-01: VulkanNN Rusted Edition

Initial Rust rewrite using PyO3 and maturin. Introduced:
- `Tensor` class with Python operator overloading (`@`, `+`)
- `from_ssd` / `new_ssd` for memory-mapped SSD tensors
- WGPU backend (later replaced in v3.3.0 by raw ash Vulkan)
- memmap2 + madvise prefetching (later replaced by io_uring)
