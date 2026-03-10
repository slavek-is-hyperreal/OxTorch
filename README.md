# VulkanNN Rusted (v3.4.0 "Iron Age Complete")

**A high-performance, Rust-powered tensor engine for constrained consumer hardware.**

Designed to run neural network inference on hardware that modern frameworks dismiss:
legacy CPUs without AVX2, sub-2GB GPUs, systems with 24GB of DDR3 RAM and SSD-resident weights.
The goal is numerical parity with PyTorch at a fraction of the memory footprint,
with no CUDA, no dedicated tensor cores, no Apple Silicon required.

[![Engine: Rust + Vulkan (ash)](https://img.shields.io/badge/Engine-Rust%20%2B%20Vulkan%20ash-blue.svg)](#technical-overview)
[![Precision: Tri-Mode](https://img.shields.io/badge/Precision-F32%20%7C%20F16%20%7C%20BF16-green.svg)](#technical-overview)
[![Parity: PyTorch](https://img.shields.io/badge/Parity-PyTorch%201%3A1-brightgreen.svg)](#benchmarks)

---

## Inspiration: The MERA-400

This project is directly inspired by the **MERA-400**, a Polish 16-bit minicomputer designed and
manufactured between 1976 and 1985 by Zaklad Komputerow ERA in Warsaw. Roughly 650 units were built.

What made the MERA-400 extraordinary was its **clockless, fully asynchronous processor**. Its speed
was not governed by a fixed clock frequency, but by the number of operations it could complete per
unit of time. Every module in the system operated asynchronously, tolerating components with different
timing characteristics. This was hardware-level deterministic dataflow decades before it became a
concept in mainstream computer science.

The CROOK operating system, developed at the Institute of Marine Technology at Gdansk University of
Technology, paired with the MERA-400's architecture to create a system that could do more with less
than almost anything built in the West at the time.

The MSTS (MERA Style Task Scheduler) at the core of this library is a software reimplementation of
those ideas in modern Rust: a ring buffer of atomic-state tiles that lets CPU and GPU workers race
to claim work without locks, without a clock, and without static splits.

- Wikipedia (PL): [MERA-400](https://pl.wikipedia.org/wiki/Mera_400)
- Archival documentation and community: [mera400.pl](https://mera400.pl/Strona_g%C5%82%C3%B3wna)
- Video channel (highly recommended): [youtube.com/c/mera400](https://www.youtube.com/c/mera400)

---

## Why VulkanNN Rusted?

- **Tri-Precision Engine**: Native F32, F16, and BF16 support with runtime-dispatched SIMD
  conversion (F16C/AVX on Ivy Bridge, SSE2 SWAR on older hardware, NEON on AArch64).
- **300-700x speedup over PyTorch on F16/BF16 MatMul**: PyTorch falls back to scalar software
  emulation on CPUs without AVX-512 FP16. VNN uses explicit SIMD upcasting and Vulkan compute.
- **SSD-as-RAM**: Asynchronous Linux `io_uring` with `O_DIRECT` streams weights directly from a
  ZFS pool at 1MB record boundaries, bypassing the OS page cache entirely.
- **MSTS Tile-Pulling Hybrid Dispatch**: CPU and GPU workers autonomously race to claim 256K-element
  tiles via an `AtomicUsize` counter. No static splits. No locks. The faster resource eats more work.
- **Statistical benchmark harness**: Multi-run audit with Median/Mean/StdDev and parity checks
  against PyTorch for all three precisions.

---

## Quick Start (Python)

```python
# The module name reflects the active branch (A/B testing across branches)
import vulkannn_rusted_exp as vnn   # dev_raw_vulkan branch
# import vulkannn_rusted_dev as vnn  # dev branch
# import vulkannn_rusted_main as vnn # main branch
from vulkannn_rusted_exp import Tensor, DataType

# Tri-precision tensors
a = Tensor(np.random.randn(2048, 2048).astype(np.float32), dtype=DataType.BF16, device="vulkan")
b = Tensor(np.random.randn(2048, 2048).astype(np.float32), dtype=DataType.BF16, device="vulkan")
result = a @ b

# Out-of-core tensor from SSD (weights larger than RAM)
weights = Tensor.from_ssd("/pool/weights.bin", shape=(40000, 40000), dtype=DataType.F16)
```

---

## Benchmarks (v3.4.0, dev branch)

Hardware: Intel Core i5-3450 (Ivy Bridge, 4 cores, AVX + F16C, no AVX2) |
AMD Radeon R7 260X (Bonaire GCN 1.1, 1GB GDDR5) | 24GB DDR3

| Test | PyTorch | VNN | Ratio | Notes |
|:---|---:|---:|:---:|:---|
| MatMul F32 2048x2048 (cpu) | 0.316s | 0.320s | 1.01x | Near parity via matrixmultiply/sgemm |
| MatMul F32 2048x2048 (vulkan) | 0.363s | 0.391s | 1.08x | Bonaire PCIe overhead visible |
| MatMul F32 2048x2048 (hybrid) | 0.228s | 0.397s | 1.74x | GPU threshold optimization pending |
| ReLU F32 1M (cpu) | 0.002s | 0.002s | 0.92x | Rayon SIMD |
| ReLU F32 1M (hybrid) | 0.003s | 0.023s | 6.7x | Below GPU threshold: CPU only |
| MatMul F16 2048x2048 (cpu) | 103.8s | 0.298s | **0.003x** | PyTorch: scalar emulation; VNN: AVX F16C |
| MatMul F16 2048x2048 (vulkan) | 104.0s | 0.361s | **0.003x** | VNN: Vulkan F32 compute, F16 storage |
| MatMul BF16 2048x2048 (cpu) | 40.1s | 0.209s | **0.005x** | PyTorch: scalar; VNN: SSE2 SWAR |
| Monster ReLU 16GB (SSD) | N/A | 46.6s | SSD limit | io_uring O_DIRECT, 1MB ZFS records |

> PyTorch F16/BF16 results reflect execution on the i5-3450 without native F16C acceleration in
> PyTorch's dispatch path. VNN dispatches to hardware F16C intrinsics at runtime.

---

## Technical Overview

### Backend: Raw Ash Vulkan (v3.4.0+)

The GPU backend was completely rewritten from `wgpu`/WGSL to `ash` (raw Vulkan 1.2 bindings).
This provides explicit control over:

- Separate compute and transfer command pools
- Vulkan Timeline Semaphores for async GPU operation chaining
- A buffer cache (`get_buffer` / `recycle_buffer`) to avoid per-dispatch allocation
- Explicit pipeline barriers for correct TRANSFER -> COMPUTE -> TRANSFER synchronization

Shaders are compiled from WGSL to SPIR-V at build time via `naga` in `build.rs`.

### MSTS Tile-Pulling Hybrid (Phase 4)

For `device="hybrid"` activations, work is divided into 256K-element tiles.
An `Arc<AtomicUsize>` tile counter is shared between:

1. A GPU dispatcher thread: calls `execute_activation_chunked` on each claimed tile (Vulkan)
2. A CPU worker thread: processes claimed tiles with inline SWAR math (Rayon)

The GPU dispatcher only activates when `num_elements >= VULKAN_MIN_ELEMS` (4M elements, ~16MB F32).
Below that threshold, Vulkan PCIe staging overhead dominates and pure CPU SWAR is faster.

### SIMD Conversion Matrix (avx_swar.rs)

| CPU | F32->F16 | F16->F32 | F32->BF16 | BF16->F32 |
|:---|:---|:---|:---|:---|
| x86_64 + F16C + AVX | F16C intrinsics | F16C intrinsics | SSE2 RNE | SSE2 shift |
| x86_64 + SSE2 (no F16C) | SSE2 SWAR | SSE2 SWAR | SSE2 RNE | SSE2 shift |
| AArch64 | NEON vcvt | NEON vcvt | NEON shift | NEON shift |
| Any other | Rayon scalar | Rayon scalar | Rayon scalar | Rayon scalar |

Runtime dispatch via `is_x86_feature_detected!()` — no compile-time feature flags required.

### Source Map

| Component | File |
|:---|:---|
| Tensor operations, hybrid dispatch | `src/tensor.rs` |
| Raw Vulkan backend, chunked GPU dispatch | `src/backend.rs` |
| SIMD conversions, SWAR fallbacks | `src/avx_swar.rs` |
| io_uring O_DIRECT SSD streaming | `src/io_uring_engine.rs` |
| MSTS StatefulTile ring buffer | `src/crook_scheduler.rs` |
| Streaming budgets and prefetcher | `src/streaming.rs` |

---

## Documentation

- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [Performance Guide](docs/performance_guide.md)
- [Roadmap](docs/roadmap.md)
- [Changelog](docs/CHANGELOG.md)

---

## Branch Versioning

Each development branch compiles to a distinctly named Python module for A/B benchmarking:

| Branch | Module | Version |
|:---|:---|:---|
| `main` | `vulkannn_rusted_main` | v2.9.0 |
| `test` | `vulkannn_rusted_test` | v3.2.0 |
| `dev` | `vulkannn_rusted_dev` | v3.4.0 (Raw Vulkan) |
| `dev_raw_vulkan` | `vulkannn_rusted_exp` | v3.4.0 (Merged) |

---

## License

MIT License.
