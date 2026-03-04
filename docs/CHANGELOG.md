# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.9.0] - 2026-03-04
### Added
- **Statistical Guard**: Integrated Coefficient of Variation (CV%) and P95 percentile tracking in benchmarks to filter system noise.
- **API Reference**: Detailed documentation of all Rust/Python bindings with source line references.
- **Hardware-Invariant Monitoring**: Switched to Ratio (VNN/PT) for regression detection.

## [2.8.19] - 2026-03-04
### Added
- **Gemma Optimization**: Achieved 0.74x Ratio on Gemma 2B and 0.82x on Gemma 3 4B.
- **Batch GEMV Submits**: Collected command buffers for concurrent processing in GEMV paths.

## [2.8.18] - 2026-03-03
### Changed
- **Zero-Overhead CPU**: Removed manual threading in favor of `matrixmultiply`'s internal pool, achieving 0.86x Ratio on CPU MatMul.
- **Refactored Hybrid Split**: Dynamic work-stealing for M-blocks between CPU and GPU.

## [2.8.17] - 2026-03-02
### Added
- **The Union**: Forced N-tiling for GEMV to enable Double Buffering across all shapes.
- **Double Buffering**: Dual weight buffers in `backend.rs` to overlap IO and Compute.

## [2.8.0] - 2026-03-01
### Added
- **Rust-First Core**: Initial release of the `vulkannn_rusted` engine, replacing the legacy Taichi-based Python core with raw WGPU/Rust.
- **SSD L3 Cache**: Memory mapping for tensors larger than system RAM.

## [Legacy]

### Added
- **PagedAttention & Context Management**: Developed `BlockTable` and `PagedKVCache` in `paged_attention.py` to virtually map non-contiguous fragments of memory dynamically token-by-token, dramatically reducing OOM errors caused by contiguous cache fragmentation.
- **KVCachePool**: Global physical memory pool backing the PagedAttention blocks via Taichi `ndarray`.
- **PagedAttention Custom Shader**: Added highly optimized `k_paged_attention_vulkan` kernel in `kernels.py` handling query dot products and physical pointer resolution across scattered block tables.
- **Layers**: New `PagedAttention` module added to `modules/layers.py` that transparently accepts normal Queries but fetches Keys/Values safely from the PagedKVCache.
- **VulkanTensorPool**: A new centralized memory suballocator in `vulkan_nn_lib/memory_pool.py` that intercepts all tensor creations to manage `maxMemoryAllocationCount` Vulkan limits and prepare for PagedAttention (KV Cache fragmentation handling).
- **Comprehensive Parity Suite**: Unified Python test architecture (`tests/parity/test_pytorch_parity_comprehensive.py`) covering structural ops, math functions, activations, linear algebra, and arithmetics across CPU, Vulkan, and SSD backends.
- `docs/CHANGELOG.md` to track project evolution.
- Experimental `debug_matmul.py` and `debug_leaky_relu.py` scripts for isolated precision tracking.
- **Memory Benchmarks**: `tests/core/benchmark_paged_attention.py` leveraging Linux `/proc/self/status` to accurately trace VM RSS optimization compared to PyTorch's naive contiguous allocation.

### Changed
- **Autograd Engine**: Integrated `VulkanTensorPool` seamlessly into `vulkan_nn_lib/tensor.py` ensuring all tensor instantiations (zeros, external, clones) route through the centralized pool allocator.
- **Git Branching Strategy**: Transitioned repository to a standard workflow (`main` for public releases, `test` for release candidates, `dev` for active development).
- **Streaming Ops Engine (SOE)**: Complete logic overhaul in `streaming_ops.py` to properly handle unary operations (`b=None`) and correct scalar vs tensor broadcasting.

### Fixed
- **Stale RAM Shadow Bug**: Resolved a critical precision issue where `tensor.load_from_numpy()` failed to update the CPU RAM shadow cache (`self.np_arr`), causing operations like bias addition to incorrectly add zeros instead of weights.
- **LeakyReLU Precision Drop**: Corrected argument mapping in `functional.py` where the `alpha` parameter for `leaky_relu` was being incorrectly passed as the secondary operand instead of the `extra` kwarg, leading to fallback to the default slope.
- **Pow Kernel Signature Mismatch**: Fixed missing Typecasting in `pow` operation within `tensor.py` where a parameter was incorrectly passed to Taichi kernel resulting in `AttributeError`.
- **SystemExit Bug**: Removed hard-coded `sys.exit()` calls from PyTest suites to ensure they report clean `PASSED` status logs without triggering pipeline failures.
- **Taichi Deprecation Warning**: Patched local virtual environment `taichi/_lib/utils.py` to prefer `locale.getencoding()` over the deprecated `locale.getdefaultlocale()` to prevent Python 3.12+ warnings.
- **AutoOptimizer Arguments**: Fixed `AttributeError` during Taichi compilation in `AutoSGD` and `AutoAdam` due to argument signature mismatch (`p.total_size` passed as parameter to kernel float argument).
- **Tensor.to_numpy() Cache Bug**: Fixed stale caching returning old references for updated Vulkan optimizers during parity training steps.
- **TensorStore Missing Directory Crash**: Fixed a recurring `FileNotFoundError` during sequential integration tests by explicitly forcing recreating cache directories `os.makedirs()` when allocating SSD tensors.

---
*Generated during Phase 1 (Memory Suballocation) sprint.*
