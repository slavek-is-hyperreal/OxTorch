# OxTorch (v3.7.0 — "The BitNet Leapfrog")

**A high-performance, Rust-powered tensor engine for constrained consumer hardware.**

Designed to run AI inference on hardware modern frameworks quietly abandoned:
legacy CPUs without AVX2, sub-2GB GPUs, systems with DDR3 RAM and SSD-resident weights.
No CUDA. No dedicated tensor cores. No Apple Silicon required.

[![Engine: Rust + Vulkan](https://img.shields.io/badge/Engine-Rust%20%2B%20Vulkan%20ash-blue.svg)](#technical-overview)
[![Precision: Quad-Mode](https://img.shields.io/badge/Precision-F32%20%7C%20F16%20%7C%20BF16%20%7C%20INT8-green.svg)](#architecture-support-matrix)
[![Drop-in: oxtorch](https://img.shields.io/badge/Drop--in-import%20oxtorch%20as%20torch-brightgreen.svg)](#one-import-drop-in)

---

## ⚡ One-Import Drop-In

OxTorch ships with a Python package called `oxtorch` that is a **transparent drop-in replacement** for PyTorch:

```python
# Before:
import torch

# After — that's the entire change:
import oxtorch as torch

# Everything below works exactly as before.
result = torch.matmul(a, b)
x = torch.relu(x)
logits = torch.nn.functional.softmax(x, dim=-1)
```

### How it works

```
import oxtorch as torch
         │
         ▼
  torch.anything(...)
         │
         ├─ OxTorch natively supports it? ──► Rust kernel (Vulkan/CPU SIMD)
         │                                    (may be 4–25x faster)
         │
         └─ Not yet implemented? ───────────► transparent fallback:
                                              tensor → numpy → real PyTorch → back
                                              (always correct, PyTorch speed)
```

**What this means in practice:**
- **Inference always works.** If OxTorch can't accelerate an op, it silently delegates to PyTorch. You will never get a `NotImplementedError`.
- **Training with autograd doesn't work yet.** `requires_grad`, `.backward()` are Sprint 7 (long-term). Inference-only code runs fine.
- **PyTorch must be installed.** The fallback mechanism requires a working `torch` installation. OxTorch accelerates — it does not replace when it can't.
- **`isinstance(x, torch.Tensor)` checks** will see `OxTorchTensor`, not `torch.Tensor`. Code that hard-codes type checks may need adjustment.

### When does OxTorch accelerate?

| Operation | Accelerated? | Notes |
|:---|:---:|:---|
| `torch.matmul` (F16/BF16) | ✅ **Yes — massively** | 4–25x faster than PyTorch on GCN/Ivy Bridge |
| `torch.matmul` (F32) | ✅ Yes | ~0.5–2x depending on size |
| `torch.matmul` (INT8) | ✅ Yes | PyTorch scalar → OxTorch AVX2 SIMD |
| `torch.relu`, `torch.gelu` | ✅ Yes (CPU) | AVX1/F16C/NEON kernels |
| `torch.sum`, `torch.softmax` | ✅ Yes (CPU) | Vectorized reductions |
| `torch.nn.functional.gelu` (INT8) | ⚠️ Fallback | PyTorch doesn't support GELU on INT8 at all — OxTorch does natively |
| `torch.nn.Module.forward()` | ✅ Works | Sub-ops either accelerated or fall back |
| `model.backward()` / autograd | ❌ Not yet | Sprint 7 |
| `torch.save` / `torch.load` | ❌ Not yet | Sprint 5 |

---

## 🚀 Getting Started

### 1. Install the Rust Compiler
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Set Up Your Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install maturin numpy torch   # torch required for parity fallback
```

### 3. Compile and Install
```bash
cd vulkannn_rusted
maturin develop --release
```

### 4. Use it

**Option A — Drop-in (recommended):**
```python
import oxtorch as torch
import numpy as np

# Your existing PyTorch code, unchanged
a = torch.from_numpy(np.random.randn(2048, 2048).astype(np.float32))
b = torch.from_numpy(np.random.randn(2048, 2048).astype(np.float32))
result = torch.matmul(a.half(), b.half())   # routes to Vulkan F16 kernel
```

> ⚠️ **Note on PYTHONPATH**: `oxtorch` is not yet pip-installable as a standalone package.
> When running scripts directly, set: `PYTHONPATH=/path/to/gaussian_room/vulkannn_rusted`

**Option B — Native API (full control):**
```python
import vulkannn_rusted as vnn
import numpy as np

a = vnn.Tensor(np.random.randn(2048, 2048).astype(np.float32), dtype=vnn.DataType.BF16, device="vulkan")
b = vnn.Tensor(np.random.randn(2048, 2048).astype(np.float32), dtype=vnn.DataType.BF16, device="vulkan")
result = a @ b   # 20x faster than PyTorch on AMD Bonaire

# Out-of-core tensor from SSD (weights larger than RAM)
weights = vnn.Tensor.from_ssd("/pool/weights.bin", shape=(40000, 40000), dtype=vnn.DataType.F16)
```

---

## Benchmarks (v3.7.0, 53 tests, AMD Radeon R7 / i5-3450)

Hardware: Intel Core i5-3450 (Ivy Bridge, AVX + F16C, no AVX2) |
AMD Radeon R7 200 Series (Bonaire GCN 1.1, 1GB GDDR5) | 24GB DDR3

**OxTorch faster in 34/53 benchmark runs (64%).**

### MatMul — OxTorch crushes PyTorch on legacy hardware

| Test | PyTorch | OxTorch | Ratio | Notes |
|:---|---:|---:|:---:|:---|
| MatMul f16 (vulkan) | 4.61s | 0.20s | **0.04x** 🚀 | PT uses scalar emulation, no AVX-512 FP16 |
| MatMul bf16 (vulkan) | 5.87s | 0.29s | **0.05x** 🚀 | Same — PT falls back to scalar |
| MatMul f32 (vulkan) | 5.34s | 0.34s | **0.06x** 🚀 | Vulkan tiled compute shader |
| MatMul int8 (cpu) | 4.63s | 0.17s | **0.04x** 🚀 | AVX2 SIMD vs PT scalar int8 |
| MatMul int8 (vulkan) | 1.13s | 0.17s | **0.15x** 🚀 | GPU compute shader |

### Element-wise ops — Mixed results

| Test | PyTorch | OxTorch | Ratio | Notes |
|:---|---:|---:|:---:|:---|
| ReLU int8 (cpu) | 3.1ms | 0.26ms | **0.085x** 🚀 | Dedicated INT8 SIMD kernel |
| ReLU f32 (cpu) | 4.0ms | 1.8ms | **0.44x** ✅ | AVX1 `vmaxps` |
| ReLU 15M f16 (hybrid) | 82.7ms | 50.8ms | **0.62x** ✅ | MSTS tile-pulling |
| ReLU 15M f32 (vulkan) | 24.3ms | 73.2ms | 3.01x ⚠️ | Vulkan PCIe overhead on 15M elems |
| ReLU 15M int8 (hybrid)| 2.9ms | 46.3ms | 15.8x ⚠️ | PCIe cost kills small INT8 tensors |

> ⚠️ Vulkan overhead: AMD Bonaire (GCN 1.1) has ~80ms PCIe round-trip cost. For tensors below
> ~4M elements, OxTorch stays on CPU. The GPU wins decisively only for large MatMuls.

---

## Architecture Support Matrix

### CPU Support (SIMD & SWAR)
| Op | `no-avx` (SSE) | `avx1` (Ivy Bridge) | `avx2` (Haswell+) | `arm_neon` |
|:---|:---:|:---:|:---:|:---:|
| **MatMul F32** | ✅ Native | ✅ Native | ✅ Native | ✅ Fallback |
| **MatMul F16** | ✅ SWAR | ✅ F16C | ✅ F16C | ✅ NEON |
| **MatMul INT8** | ✅ Scalar | ✅ SSE4.1 | ✅ AVX2 | ✅ NEON |
| **BitLinear** | ✅ Ternary | ✅ Ternary | ✅ Ternary | ✅ Ternary |
| **ReLU / GELU** | ✅ Native | ✅ AVX1 | ✅ AVX2 | ✅ NEON |
| **Softmax** | ✅ Native | ✅ Native | ✅ Native | ✅ Fallback |
| **INT8 GELU** | ✅ **OxTorch only** | ✅ **OxTorch only** | ✅ **OxTorch only** | ✅ **OxTorch only** |

### Vulkan (GPU) Support
| Op | F32 | F16 | INT8 | Ternary |
|:---|:---:|:---:|:---:|:---:|
| **MatMul** | ✅ | ✅ | ✅ | N/A |
| **Activations** | ✅ | ✅ | ✅ | N/A |
| **BitLinear** | ✅ | ✅ | ✅ | ✅ |

---

## Technical Overview

### Backend: Raw Ash Vulkan (v3.7.0+)

GPU backend uses `ash` (raw Vulkan 1.2 bindings) for explicit control over:
- Separate compute and transfer command pools
- Vulkan Timeline Semaphores for async GPU operation chaining
- Buffer cache (`get_buffer` / `recycle_buffer`) to avoid per-dispatch allocation
- Explicit pipeline barriers for TRANSFER → COMPUTE → TRANSFER synchronization

Shaders compiled from WGSL to SPIR-V at build time via `naga` in `build.rs`.

### MSTS Tile-Pulling Hybrid Dispatch

For `device="hybrid"`, work is divided into 256K-element tiles.
An `Arc<AtomicUsize>` tile counter shared between a GPU thread and CPU thread:

```
total_elements → N tiles of 256K each
tile_counter = AtomicUsize(0)

Rayon scope:
  [GPU thread]: loop { tile = tile_counter.fetch_add(1); dispatch to Vulkan }
  [CPU thread]: loop { tile = tile_counter.fetch_add(1); compute with SIMD }
```

No locks, no static splits. The faster resource eats more work.
GPU dispatched only if `num_elements >= 4M` (Bonaire PCIe latency threshold).

### SIMD Conversion Matrix

| CPU | F16↔F32 | BF16↔F32 |
|:---|:---|:---|
| x86_64 + F16C + AVX | F16C intrinsics | SSE2 RNE |
| x86_64 + SSE2 (no F16C) | SSE2 SWAR | SSE2 RNE |
| AArch64 | NEON `vcvt` | NEON shift |
| Other | Rayon scalar | Rayon scalar |

Runtime dispatch via `is_x86_feature_detected!()` — no compile-time flags.

### SSD-as-RAM (io_uring + O_DIRECT)

`Tensor.from_ssd()` streams weights via Linux `io_uring` with `O_DIRECT`, bypassing the page cache entirely at ~86 MB/s effective throughput. Tested with a 16GB Monster ReLU (4B F32 elements, 46.5s).

### Source Map

| Component | File |
|:---|:---|
| Python API & Tensor | `src/tensor/mod.rs` |
| CPU Backend | `src/cpu/mod.rs` |
| Vulkan Backend | `src/backend.rs` |
| OxTorch Drop-in Package | `oxtorch/__init__.py`, `oxtorch/tensor.py` |
| BitNet Kernels | `src/cpu/ops/bit_linear.rs` |
| SSD Streaming | `src/streaming.rs`, `src/io_uring_engine.rs` |
| MSTS Scheduler | `src/crook_scheduler.rs` |
| SIMD Kernels | `src/cpu/ops/` |

---

## Inspiration: The MERA-400

This project is directly inspired by the **MERA-400**, a Polish 16-bit minicomputer designed at Zakład Komputerów ERA, Warsaw (1976–1985). ~650 units built.

Its **clockless, fully asynchronous processor** ran at a speed governed not by a fixed clock, but by the number of operations per unit time — each module operating independently. The CROOK OS paired with it created a system that did more with less than almost anything built in the West at the time.

The **MSTS (Mera Style Tiling System)** at the core of OxTorch reimplements those ideas in modern Rust: a ring buffer of atomic-state tiles that lets CPU and GPU workers race to claim work without locks, without a clock, and without static splits.

- [MERA-400 Wikipedia (PL)](https://pl.wikipedia.org/wiki/Mera_400) | [mera400.pl](https://mera400.pl) | [YouTube](https://www.youtube.com/c/mera400)

---

## Documentation

- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [Performance Guide](docs/performance_guide.md)
- [Roadmap](docs/roadmap.md)
- [Changelog](docs/CHANGELOG.md)

---

## Why OxTorch? (The Engineering Philosophy)

*"The MERA-400 ran a distributed OS on components with varying timing characteristics in 1976. Constraints breed architecture."*

- **Tile everything.** No operation requires more VRAM/RAM than a single tile. The MSTS ring streams data through a fixed window regardless of model size.
- **Fuse ruthlessly.** Every PCIe round-trip costs time. MatMul + Bias + ReLU → one shader dispatch, not three.
- **Use what the hardware has.** Ivy Bridge has AVX1 + F16C. That's `vmaxps` for ReLU, `vcvtph2ps` for F16. No AVX2? No problem — we don't need it.
- **Asynchronous by default.** CPU workers and the Vulkan queue race to claim tiles via a single `AtomicUsize` — no locks, no static splits, the faster path eats more work.

---

## License

MIT License.
