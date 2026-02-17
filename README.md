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

# 1. Monster Scale Initialization (10GB+)
w = torch.randn(1024, 1024, 2560, requires_grad=True) 

# 2. SSD-Native Forward Pass
loss = (w * 2.0).sum()

# 3. OOM-Safe Backpropagation (SSD-Native Accumulation)
loss.backward()

print(f"Gradient on SSD: {w.grad.device}")
```

---

## 📚 Documentation (The "Ultra-Detailed" Manual)

For deep technical insights, architecture diagrams, and API references, check our documentation:

| Guide | Description |
| :--- | :--- |
| 🏗️ **[Architecture](docs/architecture.md)** | Multi-tier bridge logic, SSD Autograd, and ARAS details. |
| 📜 **[Tensor API Reference](docs/tensor_api.md)** | Deep dive into every function, from `backward` to `item`. |
| ⚡ **[Performance Guide](docs/performance_guide.md)** | VNN vs PyTorch: Performance of SSD-Native backprop. |

---

## 📁 Repository Structure

-   **[vulkan_nn_lib/](vulkan_nn_lib/)**: Core library (Shim, Kernels, ARAS engine).
-   **[docs/](docs/)**: Technical deep-dives and implementation details.
-   **[demos/splat_studio/](demos/splat_studio/)**: 3D Gaussian Splatting optimized with VNN.
-   **[tests/](tests/)**: Verification suite including **SSD Autograd 1GB tests**.

---

## 💎 Features

-   **SSD-Native Autograd**: Perform backpropagation on models that exceed RAM capacity.
-   **ARAS (Adaptive RAM-Aware Streaming)**: Processes tensors in tiles based on real-time RAM availability.
-   **Tiled Reductions**: OOM-safe `sum` and `mean` for multi-gigabyte tensors.
-   **Zero-Copy Mounting**: Initialize tensors from binary files on disk in milliseconds without reading data.

---
*Developed with 💙 for the open hardware community.*
