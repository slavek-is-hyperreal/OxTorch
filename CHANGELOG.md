# Changelog

All notable changes to the VNN project will be documented in this file.

## [2.8.0] - 2026-03-03
### Added (Rusted Ed)
- **CPU Superiority**: Achieved ~0.9x - 0.99x execution time vs PyTorch for large MatMul and ReLU operations on local RAM.
- **Async Triple-Buffering**: New 3-stage pipeline in `backend.rs` for overlapping GPU compute with SSD/RAM I/O.
- **256-Thread WGSL Shaders**: Modernized shader architecture for better hardware occupancy and stability.
- **Unified Benchmark**: Integrated `unified_benchmark.py` for continuous performance and parity monitoring.

## [2.5.0] - 2026-03-01: Phase 5 and 6: VulkanNN Rusted Ed
Introduced the native **VulkanNN Rusted Ed** library written in Rust, designed as a 1:1 "drop-in replacement" for `vulkan_nn_lib`. 
This solution removes Python interpreter bottlenecks during out-of-core operations.

### Added (Rust)
- **Module `vulkannn_rusted`**: A completely native library built using `PyO3` and the `maturin` build system.
- **WGPU Backend**: Compute shaders written in pure WGSL supporting Addition, Matrix Multiplication (MatMul), and activation functions (ReLU, Sigmoid, SiLU). Multi-dimensional WGPU workgroups breaking the 64k dispatch allocation limit, allowing for unlimited array sizes.
- **Extremely fast DMA pipeline (Tiered Memory Cache)**: 
  - **L3 Cache (Disk)**: Zero-copy integration via `memmap2`.
  - Advanced OS-level prefetching utilizing POSIX `madvise(MADV_WILLNEED)`, telling the Linux kernel to load data into RAM asynchronously via the DMA disk controller.
  - **L1 Cache (VRAM)**: Implementation of "Ping-Pong" WGPU Buffer recycling to eliminate driver latency spikes during per-operation input and output memory allocation on the graphics card.
- **Performance Tests**: Scripts `bench_table.py` and `benchmark_rust_vs_py.py` verifying arithmetic correctness against NumPy (100% Parity ✅) and operational speed.
- **Full Python Parity API**: Out-of-the-box support for logical operators (+, @) and the `Tensor(data)` class mirroring its Python counterpart.

### Changed (Python `vulkan_nn_lib` and others)
- Fixed numerical precision in the Python Taichi backend. Replaced `k_reduce_sum` which prevented incorrectly returning 0.0 in reductions involving the `float32` type.
- Integrated operations for **Kaggle Mode**. Significantly expanded the capabilities of delegating GB-threshold `MatMul` operations offline to a virtual super-machine via Kaggle API, including session resume support.
- Optimized `to_numpy()` for rapid access (fast path bypass).
- Updated CPU mode to allow holding massive tensors entirely in RAM, reducing memory footprint.

### Fixed
- Fixed `AttributeError` bugs for tensors tied to system memory (RAM-resident arrays).
- Resolved bugs involving VRAM dropping below 2GB (removed the 512MB RAM limiter for the engine on older GPU systems), optimizing maximum output capabilities via advanced budget detection.
