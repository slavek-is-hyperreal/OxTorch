# 🌌 VNN Rusted (VulkanNN Native Edition)

**"Because it SHOULD be possible to run state-of-the-art AI on your old hardware—faster than PyTorch."**

VNN Rusted is a high-performance, native C/Rust tensor engine designed for the modern open-weights era. Built with **Rust + WGPU**, it bypasses Python interpreter bottlenecks to deliver unprecedented performance on consumer CPUs and older GPUs.

---

## 🚀 Performance: The PyTorch Killer (v2.8)

In version 2.8, VNN Rusted achieved **CPU Superiority**, consistently outperforming PyTorch in key LLM operations on standard consumer hardware.

| Operation | PyTorch CPU | VNN Rusted 2.8 | Ratio | Status |
| :--- | :--- | :--- | :--- | :--- |
| **MatMul 10k** | 34.19s | **32.75s** | **0.96x** | ✅ **SUPERIORITY** |
| **ReLU 250M** | 0.32s | **0.24s** | **0.76x** | ✅ **DOMINATION** |
| **Gemma 3 Layer** | 1.40s | **1.12s** | **0.80x** | ✅ **LLM GEN READY** |

---

## 🏗️ Core Architecture (v2.8+)

VNN Rusted employs a **Software-Defined Memory Hierarchy** designed to prevent OOM (Out-of-Memory) crashes at all costs while maintaining extreme throughput.

### 🔥 Key Technologies & Optimizations:
- **Async 3-Stage Pipeline**: Overlaps SSD/RAM I/O with GPU computation using a triple-buffering system. The GPU never waits for the bus.
- **256-Thread WGSL Core**: Modernized shaders utilizing `@workgroup_size(256)` and dynamic 2D dispatch logic (64k+ barrier bypass).
- **Zero-Copy CPU Path**: Direct integration with high-performance BLAS kernels and Rayon parallel iterators, eliminating redundant allocations.
- **L3 (SSD) Virtual Tensors**: Mount models directly from disk via `memmap2` with `madvise(MADV_SEQUENTIAL)` kernel hints. SSD is your new RAM.

---

## 🛠️ Getting Started

### 1. Build & Install
VNN Rusted is a native Python extension built with `PyO3` and `maturin`.

```bash
# 1. Activate your environment
source venv/bin/activate

# 2. Enter the Rust core
cd vulkannn_rusted

# 3. Build for peak performance
maturin develop --release
```

### 2. Basic Usage
Fast, native, and drop-in compatible API:

```python
import vulkannn_rusted as vnn
import numpy as np

# Create tensors (automatically handles RAM vs SSD)
a = vnn.Tensor(np.random.rand(1024, 1024).astype(np.float32), device="cpu")
b = vnn.Tensor(np.random.rand(1024, 1024).astype(np.float32), device="cpu")

# High-speed math
c = a @ b
d = c.relu()

print(f"Result shape: {d.shape} on {d.device}")
```

### 3. Verification
Run the unified performance and parity suite:
```bash
PYTHONPATH=. python3 tests/unified_benchmark.py
```

---

## 📚 Technical Manuals
- 🏗️ **[Architecture](docs/architecture.md)**: Deep dive into async pipelines and memory tiers.
- 🌳 **[Roadmap](docs/roadmap.md)**: Future support for Gemma 3n and MatFormer.
- 🔬 **[Walkthrough](Python_Legacy/docs-python/technical_manual.md)**: Line-by-line internal logic (Classic/Legacy).

---

## 🐍 Python Legacy Version
The original Python/Taichi implementation of VNN is now archived and available in the [Python_Legacy](Python_Legacy/README-PythonLegacy.md) directory. It remains stable but is no longer the primary focus of development.

---
*Developed with 💙 for the open hardware and self-hosting community.*
