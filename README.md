# 🚀 VulkanNN Rusted (v2.9.0)

**The Iron Age of Neural Inference.**  
A high-performance, Rust-powered tensor engine designed for extreme memory efficiency and raw speed on consumer hardware. Leveraging Vulkan WGSL shaders and zero-copy SSD mapping to run models that shouldn't fit in your RAM.

[![Performance: Record-Breaking](https://img.shields.io/badge/Performance-Record--Breaking-orange.svg)](#benchmarks)
[![Engine: Rust + Vulkan](https://img.shields.io/badge/Engine-Rust%20%2B%20Vulkan-blue.svg)](#technical-deep-dive)

---

## 🏗 Why VulkanNN Rusted?

*   **🚀 Faster than PyTorch**: Up to **2.2x speedup** on MatMul 2k (0.45x Ratio) using Hybrid GPU/CPU tiling.
*   **🧠 LLM Specialist**: Optimized for Gemma 2B (0.74x Ratio) and Gemma 3 4B (0.82x Ratio) on mid-range hardware.
*   **💾 SSD-as-RAM (L3 Cache)**: Map 16GB+ weights directly to memory via `memmap2` with background prefetching.
*   **⚡ Ghost Speed**: Asynchronous command submission and double buffering to saturate PCI-e bandwidth.
*   **🛡 Statistical Guard**: Built-in regression monitoring with Coefficient of Variation (CV) tracking.

---

## 🏁 Quick Start (Python)

Minimal knowledge required. If you have `maturin` and a Vulkan-capable GPU, you're ready.

```python
from vulkannn_rusted import Tensor
import numpy as np

# 1. Create tensors (CPU, Vulkan, or Hybrid)
a = Tensor(np.random.randn(2048, 2048).astype(np.float32), device="vulkan")
b = Tensor(np.random.randn(2048, 2048).astype(np.float32), device="vulkan")

# 2. Raw Speed MatMul
res = a @ b

# 3. Memory Mapping (Load 16GB weights in 0.0s)
weights = Tensor.from_ssd("path/to/weights.bin", shape=(4096, 4096))
# background prefetching is automatic!
```

---

## 📊 Benchmarks (v2.9.0)

Tested on the **"Slavek Lab" Baseline**:
*   **CPU**: Intel(R) Core(TM) i5-3450 @ 3.10GHz (4 Cores/4 Threads)
*   **GPU**: AMD Radeon R7 200 Series (RADV BONAIRE) (Vulkan)
*   **RAM**: 23GB DDR3
*   **OS**: Linux x86_64

| Test Case | PyTorch (Avg) | **VNN Rusted (Avg)** | **Ratio (VNN/PT)** | Stability (CV%) |
|:--- |:--- |:--- |:--- |:--- |
| **MatMul 2k (Hybrid)** | 0.203s | **0.090s** | **0.45x** 🚀 | 7.5% |
| **MatMul 2k (Vulkan)** | 0.200s | **0.097s** | **0.49x** | 3.5% |
| **Gemma 2B (Layer)** | 1.859s | **1.368s** | **0.74x** | 1.2% |
| **Gemma 3 4B (Layer)** | 1.575s | **1.290s** | **0.82x** | 0.9% |
| **MatMul 10k (CPU)** | 22.58s | **22.54s** | **1.00x** | 0.8% |

---

## 🛠 Technical Deep Dive

### The Multi-Tier Cache Analogy
VNN Rusted treats your hardware like a tiered cache system:
*   **L1 (VRAM)**: Extreme speed, small capacity (1GB-2GB). Used for active tiles.
*   **L2 (System RAM)**: Medium speed, large capacity. Data is prefetched here.
*   **L3 (SSD)**: Mass storage. Weights stay here until touched.

### Key Implementation Details
1.  **Hybrid Tiling**: `src/backend.rs:276` splits MatMul work between CPU (Rayon/matrixmultiply) and GPU (wgpu/WGSL).
2.  **Double Buffering**: `src/backend.rs:329` uses dual `bufs_b` to upload the next B-tile while the GPU is still computing the current one.
3.  **Ghost Speed (Async Copy)**: `src/backend.rs:428` prevents blocking the CPU while GPU results are being transferred back to RAM.
4.  **Zero-Overhead Memory Mapping**: `src/tensor.rs:43` uses `memmap2` with `libc::madvise(MADV_SEQUENTIAL)` for hardware-level read-ahead.

---

## 📚 Documentation
*   [Architecture Deep-Dive](docs/architecture.md) - Data flows and internal structures.
*   [Performance Guide](docs/performance_guide.md) - Understanding Tiling, CV%, and Throttle.
*   [API Reference](docs/api_reference.md) - Full list of methods and classes.
*   [Changelog](docs/CHANGELOG.md) - Version history.

---

## ⚖ License
MIT License. Created by Antigravity AI for the VNN Rusted Project.
