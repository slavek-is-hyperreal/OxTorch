# Changelog

## [3.7.1] - 2026-03-22 "HPC CPU Optimization & Parallel Fix"

### Added
- **Tiled Matrix Multiplication (F16/BF16)** (`src/cpu/ops/matmul/`): Re-implemented MatMul and Linear for half-precision types with 256x256 tiling.
- **BitNet Tiered Kernels** (`src/cpu/ops/bitnet/`): Modularized BitNet logic into specialized tiers (Scalar, SWAR, SSE, AVX2, AVX512).
- **Native GGUF I2_S Loader** (`src/models/bitnet.rs`): Direct parsing of Microsoft BitNet i2_s weights with ternary offset correction.
- **Zero-Allocation F32 Conversion**: Integrated `TensorPool` into the MatMul tiling loop to eliminate temporary allocations during on-the-fly precision upcasting.
- **SIMD Transcendental Functions** (`src/cpu/ops/math_simd.rs`): Vectorized `exp`, `sigmoid`, `silu`, and `tanh` approximations for AVX2 and NEON.

### Fixed
- **Parallel Mutable Borrow (E0596)**: Resolved a critical Rayon closure capture error in `f16.rs` and `bf16.rs` by hoisting raw pointer induction (`c.as_mut_ptr()`) outside the `for_each` parallel loop.
- **`TensorPool` Visibility (E0603)**: Made the `pool` module public in `crate::tensor` to allow global access to the slab allocator.
- **Compiler Warnings**: Removed redundant `mut` declarations in `TensorPool` and SIMD kernels.
- **Release Build Stability**: Verified successful compilation and installation in `--release` mode after resolving borrow checker and visibility constraints.

### Changed
- **Documentation Audit**: Translated all core library documentation to English as part of the v3.7.1 quality sweep.

---

## [3.7.1-rc] - 2026-03-21 "MSTS Dual-Path Dispatch & TensorPool"

### Added (Pulled Forward from Sprint 4)
- **MSTS 3-Path Dispatch** (`src/tensor/msts.rs`): Three compile-time dispatch paths eliminate thread-spawn overhead for SSD tensors:
  - **Path A (Direct):** tensor < `MSTS_DIRECT_MAX` (≈3 MB for Ivy Bridge) → `mmap` read + single AVX loop. Zero atomics, zero threads.
  - **Path B (Single-thread):** < 32 MB → 1 IO worker, ring depth = 2, tile ≈ 75% L2 cache.
  - **Path C (Full CrookScheduler):** ≥ 32 MB → 2 background workers + `rayon` parallel compute.
- **`save_ssd` method** (`src/tensor/constructors.rs`): Writes tensor raw bytes to disk and returns a new SSD-mapped tensor. Exposed to Python.
- **`TensorPool` slab allocator** (`src/tensor/pool.rs`): Thread-local 6-bucket pool (4KB → 64MB) eliminates per-op `Vec<u8>` allocation overhead.
- **`unary_op_ssd`, `load_to_f32_vec_msts` Python bindings** (`src/tensor/mod.rs`).
- **MSTS SSD Benchmark Suite** (`tests/benchmarks/ssd/`): Three new benchmarks (`msts_path_a_relu_f32.py`, `msts_path_b_relu_f32.py`, `msts_path_c_relu_f32.py`) — verify correctness and report MB/s throughput for each dispatch path.

### Fixed
- **`Storage::drop` AlignmentMismatch panic**: Replaced incorrect `bytemuck::cast_vec` with manual `Vec::from_raw_parts` pointer reconstruction. No memory copy; correct alignment contract maintained.
- **`AlignedBuffer` indexing in `msts.rs`**: Added `.as_slice()` / `.as_mut_slice()` calls; `AlignedBuffer` does not implement `Index`.
- **Recursion guard in `mod.rs`**: Python wrappers now call `execute_*` (renamed internal) variants.

---

## [3.7.0] - 2026-03-19 "The BitNet Leapfrog & OxTorch Rebranding"
### Added
- **Project Rebranding**: Transitioned from `VulkanNN` to **OxTorch**.
- **BitNet (1.58-bit) Support**: Native `Ternary` (`{-1, 0, 1}`) quantization and `BitLinear` kernels.
- **Vulkan BitLinear Shader**: Custom compatible compute shader with manual bit-unpacking for legacy GPUs (AMD Bonaire).
- **Architecture Support Matrix**: Detailed CPU/GPU support tables in `README.md`.
- **Source File Deep Dive**: Comprehensive documentation of every Rust source file in `docs/architecture.md`.

### Fixed
- **F16 CPU Efficiency**: Parallelized the scalar fallback path for MatMul and Linear, providing significant speedups when hardware intrinsics aren't detected.
- **F32 Vulkan Stability**: Resolved numerical divergence in large-scale matrix multiplications.
- **BitNet Parity**: Achieved 100% bit-perfect parity between CPU and Vulkan BitNet implementations.

---

## [3.6.0] - 2026-03-14
### Added
- **Project Modularization (Phase 0)**: Split the monolithic `tensor.rs` into logical submodules (`src/tensor/`) and organized CPU kernels into `src/cpu/`.
- **Log-Softmax Support**: Added `is_log` parameter to the Vulkan and CPU softmax implementations for training stability.
- **Int8 SWAR (SIMD Within A Register)**: Parallel addition and ReLU for `int8` data on CPUs without AVX2.
- **MSTS (Mera Style Tiling System)**: Hybrid CPU/GPU/SSD dispatch with circular buffer prefetching.
- **Safe 64-bit Reductions**: Migrated all Int8/F32/F16/BF16 summation kernels to `i64` internal accumulation, ensuring bit-perfect parity for sums exceeding 4B elements.
- **SIMD Optimized Softmax**: Vectorized 3-pass kernels for AVX-512, AVX2, and SSE4.1, including a custom 256-bit Taylor series approximation for the exponential function.
- **Native PRNG**: Internal `Xoshiro256++` implementation to remove `numpy` dependency for tensor randomization.
- **Hard-Sync SPIR-V**: Explicit descriptor set pooling and sync for Vulkan 1.2 compute backend.

### Fixed
- Fixed race condition in Vulkan descriptor allocation.
- Corrected bias addition logic in CPU `linear` fallback.
- Resolved mutable borrow and type inference issues arising from modularization.

### Changed
- Refined `matrixmultiply` and `gemm` integration for 17% overall speedup in F32 CPU MatMul.
- Consolidated `#[pymethods]` into `src/tensor/mod.rs` to prevent implementation conflicts.
- **Dynamic Upcasting**: Enforced `DataType::F32` for all reduction outputs (Sum, Mean) to prevent saturation in `Int8` paths.
- Version bump to 3.6.0 "Hardware Acceleration & Modular Restructuring".

---

## [3.5.0] - 2026-03-11 "Sprint 1 — MLP Forward Pass"

### Added
- **Sprint 1 Complete**: Full functional and numerical parity with PyTorch for feedforward networks.
- **New Operations**: `mul`, `sub`, `div`, `reshape`, `view`, `squeeze`, `unsqueeze`, `flatten`, `GELU`, `LeakyReLU`, `ELU`, `Tanh`, `Clamp`, `sum`, `mean`, `max`, `min`, `softmax`, `log_softmax`.
- **Tensor Creators**: Added `Tensor.zeros`, `ones`, `full`, `rand`, `randn` static methods.
- **Tree-Reduction Shaders**: Implemented shared-memory parallel reduction for Sum/Mean/Max/Min on Vulkan.
- **Sprint 1 Performance Audit**: Comprehensive analysis of engine efficiency vs. optimization axioms.

### Changed
- All project manuals synchronized to v3.7.0 (The BitNet Leapfrog)
- Documentation sweep: Added comprehensive Rustdoc comments to all public functions in `lib.rs`, `tensor.rs`, `backend.rs`, `avx_swar.rs`, `buf_pool.rs`, `io_uring_engine.rs`, `crook_scheduler.rs`, and `streaming.rs`.

---

## [3.4.0] - 2026-03-10 "Iron Age Complete"

### Added
- **Official Raw Vulkan Backend**: The experimental `ash` rewrite from `dev_raw_vulkan` has been merged into the main `dev` branch. `wgpu` is officially retired in this release.
- **VRAM Memory Safety**: Added a strict 512MB VRAM cache budget, aggressive PRUNE capability, and "Retry-on-OOM" loop in `backend.rs`.
- **Python-Side GC**: Added explicit `gc.collect()` and `del` in benchmark suites for processing massive 15M-element tensors.

### Changed
- Stabilization of the "Iron Age" capabilities.
- `vulkannn_rusted_dev` python module now natively uses the raw Vulkan bindings.

---

## [3.3.0] - 2026-03-10 "Iron Age" (Experimental)

### Added
- **MSTS Tile-Pulling Hybrid Dispatch (Phase 4)**: Replaced the static 30/70 CPU/GPU split with a dynamic atomic tile counter.
- **`execute_activation_chunked`**: New `backend.rs` API to support the tile-pulling hybrid dispatch.
- **GPU Dispatch Threshold (`VULKAN_MIN_ELEMS = 4M`)**: Vulkan PCIe staging overhead threshold routing.
- **Cross-Platform SIMD Fallback Chain (`avx_swar.rs`)**: Complete rewrite with runtime dispatch (F16C+AVX -> SSE2 SWAR -> AArch64 NEON -> scalar).
- **Branch-Specific Module Naming**: `vulkannn_rusted_main`, `_test`, `_dev`, `_exp`.
- **Branch-Aware Benchmark Import**: `unified_benchmark.py` dynamic import.
- **Raw `ash` Vulkan Backend**: Complete rewrite from `wgpu`/WGSL to `ash` (raw Vulkan 1.2).
- **Out-of-Core `io_uring` Engine**: Bypasses the Linux VFS page cache with `O_DIRECT`.
- **Mera Style Tiling System (MSTS)**: Lockless ring buffer inspired by MERA-400 CROOK OS.
- **15M Element ReLU Benchmark**.

### Fixed
- **F16C Detection**: Corrected i5-3450 F16C capability status.
- **SSSE3 Cleanup**.

### Changed
- `unary_op` hybrid device path update.

---

## [3.2.0] - 2026-03-09 "Valkyrie"

### Added
- **Tri-Precision Engine**: Native F32, F16, BF16 support.
- **Statistical benchmark harness**.
- **Session duration tracking**.
- **Documentation overhaul**.

### Fixed
- PyTorch F16/BF16 parity tolerance tuning.
- Median used as primary metric.

---

## [2.9.0] - 2026-03-04

### Added
- Coefficient of Variation (CV%) and P95 percentile tracking.
- API Reference documentation.
- Hardware-invariant Ratio (OxTorch/PT).

---

## [2.8.0] - 2026-03-03

### Added
- CPU near-parity with PyTorch for large RAM-resident MatMul/ReLU.
- Unified benchmark (`unified_benchmark.py`).
- WGSL compute shaders (retired).

### Changed
- Repository restructured.
- All Rust compiler warnings eliminated.

---

## [2.5.0] - 2026-03-01: OxTorch Edition

Initial Rust rewrite using PyO3 and maturin.
- `Tensor` class with Python operator overloading.
- `from_ssd` / `new_ssd` for SSD tensors.
- WGPU backend (retired).
- memmap2 + madvise (retired).
