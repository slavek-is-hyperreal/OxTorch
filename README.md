# 🌌 VNN Legacy Edition (VulkanNN)

**"1:1 PyTorch Replacement for Hardware-Limited Systems"**

VNN is a high-performance tensor library designed to bridge the gap between massive AI models and aging or resource-constrained hardware. It treats your SSD as a first-class memory tier, allowing you to run and train 100GB+ models on systems with as little as 1GB of RAM.

## Filozofia Projektu (Core Philosophy)

PyTorch is built for speed on high-end GPUs. **VNN is built for stability on everything else.**
- **Memory-First**: We prioritize "not crashing" over raw FLOPS.
- **Hardware-Agnostic**: Vulkan support means acceleration on Intel UHD, old Radeons, and NVIDIA alike.
- **Universal Scale**: If it fits on your disk, it runs in VNN.

---

## 🚀 Quick Start (Drop-in Replacement)

VNN is designed to be a transparent replacement for PyTorch in your existing scripts.

```python
import vulkan_nn_lib.torch_shim as torch

# This tensor is 4GB, but it won't crash your 2GB RAM system.
# VNN will automatically mount it to SSD using ARAS streaming.
x = torch.randn(1024, 1024, 1024) 
y = torch.exp(x)
print(y[0, 0, 0])
```

---

## 📚 Documentation (The "Ultra-Detailed" Manual)

For deep technical insights, architecture diagrams, and API references, check our documentation:

| Guide | Description |
| :--- | :--- |
| 🏗️ **[Architecture](docs/architecture.md)** | Multi-tier bridge logic, SSD streaming, and ARAS details. |
| 📜 **[Tensor API Reference](docs/tensor_api.md)** | Deep dive into every function, from `permute` to `masked_fill`. |
| ⚡ **[Performance Guide](docs/performance_guide.md)** | VNN vs PyTorch: When to use which and how to maximize throughput. |

---

## 📁 Repository Structure

-   **[vulkan_nn_lib/](vulkan_nn_lib/)**: Core library (Shim, Kernels, ARAS engine).
-   **[docs/](docs/)**: Technical deep-dives and implementation details.
-   **[demos/splat_studio/](demos/splat_studio/)**: 3D Gaussian Splatting optimized with VNN.
-   **[tests/](tests/)**: PyTorch API parity verification suite.

---

## 💎 Features

-   **ARAS (Adaptive RAM-Aware Streaming)**: Processes tensors in tiles based on real-time RAM availability.
-   **SSD-Native Ops**: Matrix Multiplication and Element-wise operations that never touch RAM in bulk.
-   **Zero-Copy Mounting**: Initialize tensors from binary files on disk in milliseconds without reading data.

---
*Developed with 💙 for the open hardware community.*
