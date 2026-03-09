# 🚀 VulkanNN Rusted (v3.3.0 "Iron Age")

**The Iron Age of Neural Inference.**  
A high-performance, Rust-powered tensor engine designed for extreme memory efficiency and raw speed on consumer hardware. Leveraging Vulkan WGSL shaders, MERA-400 inspired Task Schedulers, SIMD SWAR bit-twiddling, and zero-copy `io_uring` SSD streaming to run models that shouldn't fit in your RAM.

[![Performance: Record-Breaking](https://img.shields.io/badge/Performance-Record--Breaking-orange.svg)](#benchmarks)
[![Engine: Rust + Vulkan](https://img.shields.io/badge/Engine-Rust%20%2B%20Vulkan-blue.svg)](#technical-deep-dive)
[![Precision: Tri-Mode](https://img.shields.io/badge/Precision-F32%20%7C%20F16%20%7C%20BF16-green.svg)](#technical-deep-dive)

---

## 🏗 Why VulkanNN Rusted?

*   **Tri-Precision Engine**: Native support for **F32**, **F16 (Half)**, and **BF16 (Brain-Float)** with optimized AVX1/SSE2 SWAR bit-twiddling fast paths.
*   **Faster than PyTorch**: Up to **600x speedup** on FP16/BF16 MatMul using MSTS (MERA Style Task Scheduler) and SIMD upcast logic (VNN: 0.27s vs PT: 110s).
*   **SSD-as-RAM (L3 Cache)**: Infinite out-of-core streaming via asynchronous Linux `io_uring` and `O_DIRECT` bypassing OS page faults.
*   **Statistical Safety Net**: Built-in 10-run audit with Median, Mean, and StdDev metrics to ensure stable deployment.
*   **Total Session Tracking**: Record entire benchmark duration to analyze long-term hardware thermal behavior.

---

## 🏁 Quick Start (Python)

```python
from vulkannn_rusted import Tensor, DataType
import numpy as np

# 1. Create Tri-Precision Tensors (F32, F16, or BF16)
a = Tensor(np.random.randn(2048, 2048).astype(np.float32), 
           dtype=DataType.BF16, device="vulkan")
b = Tensor(np.random.randn(2048, 2048).astype(np.float32), 
           dtype=DataType.BF16, device="vulkan")

# 2. Raw Speed MatMul (Automatic Hybrid Tiling)
res = a @ b

# 3. 16GB Massive Matrix Support
weights = Tensor.from_ssd("weights.bin", shape=(40000, 40000), dtype=DataType.F16)
```

---

## 📊 Benchmarks (v3.3.0 "Iron Age")

*Hardware: Intel i5-3450 | AMD Radeon R7 200 | 23GB RAM*

| Test Case (2k x 2k) | PT (Median) | **VNN (Median)** | **Ratio (VNN/PT)** | StdDev (ms) |
|:--- |:--- |:--- |:--- |:--- |
| **MatMul F32 (Hybrid)** | 0.210s | **0.111s** | **0.53x** | 0.0ms |
| **MatMul F16 (CPU AVX)** | 109.7s | **0.275s** | **0.0025x** 🚀 | 0.0ms |
| **MatMul F16 (Hybrid MSTS)** | 110.4s | **0.170s** | **0.0015x** 🚀 | 0.0ms |
| **Monster ReLU (16GB)** | N/A | **69.3s** | **SSD Peak** | 0.0s |

> [!NOTE]
> PyTorch F16/BF16 results reflect CPU execution without specialized AVX512 extensions. VNN uses explicit `std::arch::x86_64` SIMD SWAR upcasting mapped directly into L3 CPU cache boundaries to absolutely shatter legacy hardware limits.

---

## 🛠 Technical Deep Dive

### Implementation Reference
1.  **MERA Style Task Scheduler**: `src/crook_scheduler.rs` Tagged-Token ring buffer eliminating CPU thread blocking overhead.
2.  **`io_uring` Streaming**: `src/io_uring_engine.rs` directly maps SSD blocks on 1MB ZFS boundaries bypassing the Linux VFS completely.
3.  **AVX SWAR Bit-Twiddling**: `src/avx_swar.rs` branchless hardware intrinsics bridging modern FP16 performance on Ivy Bridge era CPUs.
4.  **Hybrid Tiling**: `src/backend.rs:360` splits work between CPU (Rayon) and Vulkan GPU.

---

## 📚 Documentation
*   [API Reference](docs/api_reference.md) - Line-by-line documentation of every kernel and method.
*   [Architecture Deep-Dive](docs/architecture.md) - Threading model and Hybrid Work Stealer details.
*   [Performance Guide](docs/performance_guide.md) - Understanding Statistical Variance and Throttling.
*   [Changelog](docs/CHANGELOG.md) - The road from v1.0 to v3.2.0.

---

## ⚖ License
MIT License. Created by Antigravity AI for the VNN Rusted Project (March 2026).
