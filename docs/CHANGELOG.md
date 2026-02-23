# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **VulkanTensorPool**: A new centralized memory suballocator in `vulkan_nn_lib/memory_pool.py` that intercepts all tensor creations to manage `maxMemoryAllocationCount` Vulkan limits and prepare for PagedAttention (KV Cache fragmentation handling).
- **Comprehensive Parity Suite**: Unified Python test architecture (`tests/parity/test_pytorch_parity_comprehensive.py`) covering structural ops, math functions, activations, linear algebra, and arithmetics across CPU, Vulkan, and SSD backends.
- `docs/CHANGELOG.md` to track project evolution.
- Experimental `debug_matmul.py` and `debug_leaky_relu.py` scripts for isolated precision tracking.

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

---
*Generated during Phase 1 (Memory Suballocation) sprint.*
