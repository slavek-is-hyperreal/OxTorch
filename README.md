# OxTorch (v3.7.1 — "MSTS Dual-Path Dispatch & TensorPool")

**Run modern AI inference on hardware that PyTorch left behind.**

OxTorch is a Rust tensor engine built for machines that are too slow, too old, or have too little RAM for mainstream frameworks.
It streams model weights from SSD tile-by-tile (never loading the full model into RAM),
pushes compute to whatever GPU the machine has via raw Vulkan, and falls back to hand-tuned SIMD for everything else.

- **No CUDA.** Works on any Vulkan-capable GPU, including decade-old AMD/Intel cards.
- **No RAM limit.** Model weights live on SSD and stream through an 8MB ring buffer.
- **No code changes.** `import oxtorch as torch` — existing PyTorch inference scripts run unchanged.

[![Engine: Rust + Vulkan](https://img.shields.io/badge/Engine-Rust%20%2B%20Vulkan%20ash-blue.svg)](#technical-overview)
[![Precision: Quad-Mode](https://img.shields.io/badge/Precision-F32%20%7C%20F16%20%7C%20BF16%20%7C%20INT8-green.svg)](#architecture-support-matrix)
[![Drop-in: oxtorch](https://img.shields.io/badge/Drop--in-import%20oxtorch%20as%20torch-brightgreen.svg)](#one-import-drop-in)

---

## ⚡ One-Import Drop-In

OxTorch ships a Python package called `oxtorch` that replaces PyTorch at the import level.
Ops that OxTorch has implemented natively run in Rust (faster). Ops it hasn't implemented yet
fall back silently to real PyTorch — you never hit a `NotImplementedError`.

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
- **PyTorch must be installed.** The fallback mechanism requires a working `torch` installation.
- **`isinstance(x, torch.Tensor)` checks** will see `OxTorchTensor`, not `torch.Tensor`. Code that hard-codes type checks may need adjustment.

### When does OxTorch accelerate?

| Operation | Accelerated? | Notes |
|:---|:---:|:---|
| `torch.matmul` (F16/BF16) | ✅ **Yes — massively** | ~400–780x faster on CPUs without AVX-512 |
| `torch.matmul` (F32) | ✅ Yes | ~0.5–2x depending on size |
| `torch.matmul` (INT8) | ✅ Yes | ~6.5–10x faster via AVX2 SIMD |
| `torch.relu`, `torch.gelu` | ✅ Yes (CPU) | AVX1/F16C/NEON kernels |
| `torch.sum`, `torch.softmax` | ✅ Yes (CPU) | Vectorized reductions |
| `torch.nn.functional.gelu` (INT8) | ✅ **OxTorch only** | PyTorch has no native INT8 GELU kernel |
| `torch.nn.functional.softmax` (INT8) | ✅ **OxTorch only** | Same — OxTorch dequantizes internally |
| `torch.nn.Module.forward()` | ✅ Works | Sub-ops accelerated or fall back |
| `model.backward()` / autograd | ❌ Not yet | Sprint 7 |
| `torch.save` / `torch.load` | ❌ Not yet | Sprint 5 |
| `tensor.save_ssd(path)` | ✅ **Yes — SSD streaming** | Writes to disk, returns MSTS-backed tensor |

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
pip install maturin numpy torch   # torch required for fallback mechanism
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

a = vnn.Tensor(data=np.random.randn(2048, 2048).astype(np.float32), dtype=vnn.DataType.BF16, device="vulkan")
b = vnn.Tensor(data=np.random.randn(2048, 2048).astype(np.float32), dtype=vnn.DataType.BF16, device="vulkan")
result = a @ b   # 20x faster than PyTorch on AMD Bonaire

# Out-of-core tensor from SSD (weights larger than RAM)
weights = vnn.Tensor.from_ssd("/pool/weights.bin", shape=(40000, 40000), dtype=vnn.DataType.F16)
```

---

## Benchmarks (v3.7.0, AMD Radeon R7 / i5-3450)

Hardware: Intel Core i5-3450 (Ivy Bridge, AVX + F16C, no AVX2) |
AMD Radeon R7 200 Series (Bonaire GCN 1.1, ~1GB GDDR5) | 24GB DDR3

> ⏳ **Note**: Benchmark results are pending the latest full run. Results below are from the most recent verified session.

### MatMul — OxTorch crushes PyTorch on legacy hardware

| Test | PyTorch | OxTorch | Ratio | Notes |
|:---|---:|---:|:---:|:---|
| MatMul F16 (vulkan) | 120.9s | 0.17s | **~0.0014x** 🚀 | ~780x faster |
| MatMul BF16 (vulkan) | 68.9s | 0.17s | **~0.0025x** 🚀 | ~400x faster |
| MatMul F16 (cpu) | 132.6s | 0.17s | **~0.0013x** 🚀 | F16C intrinsics vs PT scalar |
| MatMul INT8 (cpu) | 1.01s | 0.15s | **~0.15x** 🚀 | 6.5x faster vs PT scalar |

### Activations — Mixed results (PCIe overhead)

| Test | PyTorch | OxTorch | Ratio | Notes |
|:---|---:|---:|:---:|:---|
| ReLU INT8 (cpu) | 3.1ms | 0.26ms | **0.085x** 🚀 | Dedicated INT8 SIMD kernel |
| ReLU F32 (cpu) | 4.0ms | 1.8ms | **0.44x** ✅ | AVX1 `vmaxps` |
| ReLU 15M F16 (hybrid) | 82.7ms | 50.8ms | **0.62x** ✅ | MSTS tile-pulling |
| ReLU 15M F32 (vulkan) | 24.3ms | 73.2ms | 3.01x ⚠️ | Vulkan PCIe overhead on 15M elems |

> ⚠️ Vulkan overhead: AMD Bonaire (GCN 1.1) has ~80ms PCIe round-trip cost. For tensors below
> ~4M elements, OxTorch stays on CPU. GPU wins decisively only for large MatMuls.

---

## Architecture Support Matrix

### CPU Support (SIMD & SWAR)
| Op | `no-avx` (SSE) | `avx1` (Ivy Bridge) | `avx2` (Haswell+) | `arm_neon` |
|:---|:---:|:---:|:---:|:---:|
| **MatMul F32** | ✅ sgemm | ✅ sgemm | ✅ sgemm | ✅ gemm |
| **MatMul F16** | ✅ SWAR | ✅ F16C | ✅ F16C | ✅ NEON |
| **MatMul INT8** | ✅ Scalar | ✅ SSE4.1 | ✅ AVX2 | ✅ NEON |
| **BitLinear** | ✅ Ternary | ✅ Ternary | ✅ Ternary | ✅ Ternary |
| **ReLU / GELU** | ✅ Scalar | ✅ AVX1 | ✅ AVX2 | ✅ NEON |
| **INT8 GELU** | ✅ LUT | ✅ LUT | ✅ LUT | ✅ LUT |
| **INT8 Softmax** | ✅ dequant | ✅ dequant | ✅ dequant | ✅ dequant |

### Vulkan (GPU) Support
| Op | F32 | F16 | INT8 | Ternary |
|:---|:---:|:---:|:---:|:---:|
| **MatMul** | ✅ | ✅ | ✅ | N/A |
| **Activations** | ✅ | ✅ | ✅ | N/A |
| **Elementwise** | ✅ | ✅ | ✅ | N/A |
| **BitLinear** | ✅ | ✅ | ✅ | ✅ |

---

## Technical Overview

### Backend: Raw Ash Vulkan (v3.7.0+)

GPU backend uses `ash` (raw Vulkan 1.2 bindings) for explicit control over:
- Separate compute and transfer command pools
- Vulkan Timeline Semaphores for async GPU operation chaining
- 1GB VRAM pool with buffer cache (`get_buffer` / `recycle_buffer`)
- Explicit pipeline barriers for TRANSFER → COMPUTE → TRANSFER synchronization

Shaders compiled from WGSL/GLSL to SPIR-V at build time via `naga` in `build.rs`.

### MSTS Tile-Pulling — 3-Path Dispatch (v3.7.1+)

For `device="ssd"`, dispatch is chosen automatically based on tensor size:

| Path | Size | Threads | Strategy |
|:---|:---|:---:|:---|
| **A — Direct** | `< ~3 MB` | 0 | `mmap read_exact` + single AVX loop. Zero thread overhead. |
| **B — Single-thread** | `< 32 MB` | 1 | 1 IO worker, L2-resident tile, ring depth = 2. |
| **C — Full CrookScheduler** | `≥ 32 MB` | 2 | 2 workers + `rayon` parallel. SATA DMA saturated. |

Thresholds are compiled in by `build.rs` reading `/sys/devices` L2/L3 values (see Sprint 6 in `binary_distribution.md`).

```
total_elements → determine path A/B/C

Path C (≥32 MB):
  tile_counter = Arc<AtomicUsize>(0)
  [IO read worker 1]:  loop { claim tile; io_uring read from SSD }
  [IO write worker 2]: loop { claim tile; io_uring write to result SSD }
  [Rayon compute]:     loop { claim tile; AVX relu/gelu/... }
```

No locks, no static splits. The TensorPool slab allocator recycles tile buffers between ops.
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

`Tensor.from_ssd()` streams weights via Linux `io_uring` with `O_DIRECT`, bypassing the page cache at ~86 MB/s effective throughput. Tested with a 16GB Monster ReLU (4B F32 elements).

### Source Map

| Component | Location |
|:---|:---|
| Python API & Tensor | `src/tensor/mod.rs` |
| CPU Backend | `src/cpu/` |
| Vulkan Backend | `src/backend.rs` |
| OxTorch Drop-in Package | `oxtorch/__init__.py`, `oxtorch/tensor.py` |
| BitNet Kernels | `src/cpu/ops/bit_linear.rs` |
| SSD Streaming | `src/streaming.rs`, `src/io_uring_engine.rs` |
| MSTS Tile System | `src/crook_scheduler.rs`, `src/tensor/msts.rs` |
| SIMD Kernels | `src/cpu/ops/` |

---

## Inspiration: The MERA-400

This project is directly inspired by the **MERA-400**, a Polish 16-bit minicomputer designed at
*Zakład Komputerów Fabryki Mierników i Komputerów ERA*, Warsaw (produced 1976–1987, ~656 units built).

Its processor was **asynchronously sequential** — its speed was defined not by a clock frequency
but by the number of operations completed per unit of time (~400 000 ops/s).
The CROOK OS running on it managed preemptive multitasking on hardware that had no atomic compare-and-swap instruction.

The **MSTS (Mera Style Tiling System)** at the core of OxTorch reimplements those ideas in modern Rust:
a ring buffer of atomic-state tiles that lets CPU and GPU workers race to claim work without locks,
without a fixed clock, and without static splits.

- [MERA-400 Wikipedia (PL)](https://pl.wikipedia.org/wiki/Mera_400) | [mera400.pl](https://mera400.pl)

---

## Documentation

- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [How We Test](docs/how_we_test.md)
- [Performance Guide](docs/performance_guide.md)
- [Roadmap](docs/roadmap.md)
- [Changelog](docs/CHANGELOG.md)
- [Implementation Guides](docs/implementation_guides.md)

---

## Why OxTorch?

PyTorch requires CUDA for serious GPU work and loads full model weights into RAM.
On a machine with a 2015 AMD GPU and 8 GB RAM, running a 7B model means either renting cloud compute or not running it at all.

OxTorch attacks this differently:

- **Tile everything.** Weights stream from SSD through an 8 MB ring buffer. A 70B model needs 8 MB of working RAM, not 140 GB.
- **Use what the hardware has.** Ivy Bridge has AVX1 + F16C. That's enough for fast F16 matmul — 400–780x faster than PyTorch's scalar fallback.
- **GPU when it helps, CPU when it doesn't.** PCIe round-trips cost ~80 ms on a 2014 AMD card. OxTorch measures this and routes accordingly — no GPU dispatch under 4M elements.
- **Asynchronous by default.** CPU and GPU race over tiles via a single `AtomicUsize` — no locks, no static splits, the faster resource eats more work.

*Inspired by the MERA-400 — a Polish 16-bit minicomputer (1976, ~656 units built) whose asynchronously sequential processor ran at 400 000 ops/s without a clock signal.*

---

## License

MIT License.
