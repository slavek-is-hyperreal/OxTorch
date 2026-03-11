# VulkanNN Rusted (v3.5.0 "Sprint 1 — MLP Forward Pass")

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
| MatMul F32 2048x2048 (cpu) | ~0.2x s | 0.211s | ~1.0x | Near parity via matrixmultiply/sgemm |
| MatMul F32 2048x2048 (vulkan) | ~0.2x s | 0.091s | ~0.45x | Vulkan compute |
| MatMul F32 2048x2048 (hybrid) | ~0.2x s | 0.089s | ~0.45x | GPU threshold optimization |
| ReLU F32 1M (cpu) | 0.002s | 0.010s | 5.0x | Rayon SIMD |
| ReLU F32 1M (hybrid) | 0.001s | 0.011s | 11.0x | Below GPU threshold: CPU only |
| MatMul F16 2048x2048 (cpu) | 108.0s | 0.280s | **0.002x** | PyTorch: scalar emulation; VNN: AVX F16C |
| MatMul F16 2048x2048 (vulkan) | 106.0s | 0.250s | **0.002x** | VNN: Vulkan F32 compute, F16 storage |
| MatMul BF16 2048x2048 (cpu) | 40.5s | 0.230s | **0.005x** | PyTorch: scalar; VNN: SSE2 SWAR |
| Monster ReLU 16GB (SSD) | N/A | 45.7s | SSD limit | io_uring O_DIRECT, 1MB ZFS records |

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
| `main` | `vulkannn_rusted_main` | v3.4.0 (Iron Age Complete) |
| `test` | `vulkannn_rusted_test` | v3.4.0 |
| `dev` | `vulkannn_rusted_dev` | v3.4.0 |
| `dev_raw_vulkan` | `vulkannn_rusted_exp` | v3.4.0 (Merged into dev/main) |

---

## Plans — Becoming a Drop-In PyTorch Replacement

> *"The MERA-400 ran a distributed operating system on components with varying timing  
> characteristics in 1976. Constraints breed architecture."*

### The Vision

The end goal of VulkanNN Rusted is simple to state, and hard to build:

**Change one line of code. Everything else works.**

```python
# Old code (requires modern GPU, CUDA, AVX2):
import torch

# New code (runs on your 10-year-old machine, faster):
import vulkannn as torch
```

On **new hardware** — fast, using Vulkan compute on whatever GPU is available.  
On **old hardware** — faster than PyTorch, which penalizes legacy CPUs with scalar fallbacks
for F16/BF16. VulkanNN uses hand-crafted AVX1/F16C/SSE2 SWAR intrinsics and the Bonaire GPU
to do what PyTorch cannot.

This is not about replacing PyTorch for research. It's about making AI inference accessible
on the billions of machines that modern frameworks have quietly abandoned.

---

### Current PyTorch Parity Stage (March 2026)

We have officially **completed Sprint 1**. This means VulkanNN currently possesses **100% functional correspondence** with PyTorch for the operations required to execute an **MLP (Multi-Layer Perceptron) Forward Pass**:

- **Core Tensors**: Full parity for `shape`, `dtype` (F32/F16/BF16), and zero-copy `view`/`reshape`.
- **Primary Operators**: `@` (MatMul), `+`, `-`, `*`, `/` (Elementwise) are fully implemented across all 3 modes (CPU, Vulkan, Hybrid).
- **Activations**: 1:1 behavioral match for `ReLU`, `GELU`, `Sigmoid`, `SiLU`, `Tanh`, `ELU`, and `LeakyReLU`.
- **Reductions**: Precise parity for `sum`, `mean`, `max`, and `min` (including dimension-specific reductions).
- **Classifiers**: Functional parity for numerically stable `softmax` and `log_softmax`.
- **Creators**: Native static methods `zeros()`, `ones()`, `full()`, `rand()`, and `randn()`.

Every operation is rigorously verified via our [Tri-Precision Statistical Audit](docs/performance_guide.md) against `torch.nn.functional` to guarantee that output tensors match PyTorch internals within floating-point epsilon.

For a detailed breakdown of remaining gaps (e.g., Batch MatMul for Transformers), see our [Full PyTorch Gap Analysis](docs/roadmap.md).

### How We're Getting There — 7 Sprints

We are building this bottom-up, from the operations that matter most for running models today,
to the full training stack at the end. Each sprint is a usable milestone on its own.

| Sprint | What it enables | Key ops |
|--------|----------------|---------|
| **1 — MLP Forward Pass** | Any feedforward network works | `mul`, `div`, `sub`, `reshape`, `GELU`, `softmax`, `sum`/`mean` |
| **2 — Transformer Inference** | LLaMA/Mistral/Phi on your old machine | `bmm`, `F.linear`, `LayerNorm`, `RMSNorm`, `embedding`, attention, `cat`/`split` |
| **3 — CNN & Vision** | ResNet/ViT/EfficientNet | `conv2d` (Winograd), `pool`, `batch_norm`, full math elementwise |
| **4 — Performance** | Beat PyTorch on modern HW too | Fused kernels (MatMul+Bias+ReLU in 1 dispatch), AVX1 `vmaxps`, FlashAttention-style tiling |
| **5 — Full API Ergonomics** | True drop-in, 1 import change | `Tensor.to(dtype/device)`, broadcasting, `torch.tensor()`, `torch.save/load` |
| **6 — Quantization** | Q4/Q8 models that won't fit in RAM | GGUF, INT8/INT4, on-GPU dequantization in SPIR-V |
| **7 — Training (Long Term)** | Full training loop on old hardware | Autograd, SGD/Adam, loss functions, gradient streaming via MSTS |

---

### The Engineering Philosophy

Every sprint follows the same principles drawn from the MERA-400, CDC 6600, and Cray-1:

- **Tile everything.** No single operation should require more VRAM or RAM than the tile size.
  The MSTS ring buffer streams data through a fixed memory window regardless of model size.

- **Fuse ruthlessly.** Every PCIe round-trip between CPU and GPU costs time.
  Inspired by the Cray-1's vector chaining, intermediate results stay in GPU registers —
  MatMul + Bias + ReLU becomes one shader dispatch, not three.

- **Use what the hardware has.** Ivy Bridge has AVX1 + F16C.
  That's `vmaxps` for ReLU, `vcvtph2ps` for F16 conversion, SSE2 SWAR for BF16.
  No AVX2? No problem — we don't need it.

- **Asynchronous by default.** CPU workers and the Vulkan queue race to claim tiles
  via a single `AtomicUsize` counter — no locks, no static splits, the faster path eats more work.

Full technical detail: [docs/roadmap.md](docs/roadmap.md) | [docs/experminental_plans/deep_research_on_optimization.md](docs/experminental_plans/deep_research_on_optimization.md)

---

## License

MIT License.
