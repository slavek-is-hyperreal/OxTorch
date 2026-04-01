# OxTorch (v8.2 — "Iron Age: 2D Stride-Aware Tiling")

**Run modern AI inference on hardware that PyTorch left behind.**

OxTorch is a Rust tensor engine built for machines that are too slow, too old, or have too little RAM for mainstream frameworks. 
It streams model weights from SSD tile-by-tile (never loading the full model into RAM), pushes compute to whatever GPU the machine has via raw Vulkan, and falls back to hand-tuned SIMD for everything else.

- **No CUDA.** Works on any Vulkan-capable GPU (AMD GCN+, Intel HD 500+, NVIDIA 900+).
- **No RAM limit.** Weights stream from SSD via a hardware-tuned ring buffer (adaptive tiles) and `io_uring`.
- **No code changes.** `import oxtorch as torch` — existing PyTorch inference scripts run unchanged.

> [!IMPORTANT]
> **V8.2 "Iron Age" Status**: The Vulkan backend is now **2D Stride-Aware**. Numerical divergence in MatMul (parity drift ~244) has been resolved via native SPIR-V stride indexing. OxTorch now supports transposed and sliced tensors directly on GPU without CPU-side copies.

---

## ⚡ One-Import Drop-In

OxTorch ships a Python package called `oxtorch` that replaces PyTorch at the import level. Ops that OxTorch has implemented natively run in Rust (faster). Ops it hasn't implemented yet fall back silently to real PyTorch — you never hit a `NotImplementedError`.

```python
import oxtorch as torch

# Everything below works exactly as before.
a = torch.randn(2048, 2048, dtype=torch.bfloat16)
b = torch.randn(2048, 2048, dtype=torch.bfloat16)
result = torch.matmul(a, b) # 400x faster than PT on non-AVX512 CPUs
```

---

## 🚀 Performance (v3.8.1-rc, Ivy Bridge i5-3450)

OxTorch specializes in **Large-Vector SIMD** and **Asynchronous I/O**.

| Operation | Acceleration | Why? |
|:---|:---:|:---|
| **MatMul F16/BF16** | **400x – 780x** 🚀 | Native F16C/SSE2 vs PyTorch scalar emulation (no AVX-512). |
| **Linear BF16** | **26x** 🚀 | Optimized SIMD Core + Rayon parallelism. |
| **GELU/ReLU** | **2x – 4x** ✅ | AVX1/NEON kernels + MSTS Tiling. |
| **SSD Streaming** | **∞** 💎 | Processes 100GB+ tensors on 8GB RAM via MSTS v2. |

---

## 🛠️ Technical Overview: Iron Age (v8.2)

Version 8.2 introduces **2D Stride-Aware Tiling**. The backend now handles memory layout metadata natively.

1.  **CrookScheduler**: A triple-buffered ring of 8MB tiles.
2.  **Bitmask Barrier**: A multi-stream handshake (`A_ready | B_ready`) that allows sources to load in parallel.
3.  **Global Capacitor**: A massive RAM reservoir (50% RAM) that proactively prefetches SSD data via `io_uring`.
4.  **SIMD Auto-Dispatch**: Runtime detection of AVX2, AVX1, SSE2, and NEON.

---

## 📚 Documentation Index

For full developer guides and architecture specs, see the **[Documentation Index](docs/doc_index.md)**.

*   **[Core Architecture](docs/architecture.md)**: Decoding the Unified Pipeline.
*   **[Vulkan Internals](docs/vulkan_internals.md)**: Ash, Tiling, and 2D Strides.
*   **[API Reference](docs/api_reference.md)**: Native and Python interface documentation.
*   **[Performance Guide](docs/performance_guide.md)**: How we achieve 400x speedups on Ivy Bridge.

---

## License
MIT License. Inspired by the **MERA-400** — a Polish 16-bit minicomputer (1976).
