# 🌌 VNN Legacy Edition (VulkanNN)

**"1:1 PyTorch Replacement for Hardware-Limited Systems"**

VNN is a high-performance tensor library designed to bridge the gap between massive AI models and aging or resource-constrained hardware. It treats your SSD as a first-class memory tier, allowing you to run and train 100GB+ models on systems with as little as 1GB of RAM.

## Core Philosophy

PyTorch is built for speed on high-end GPUs. **VNN is built for stability on everything else.**
- **Memory-First**: We prioritize "not crashing" over raw FLOPS.
- **Hybrid Backend**: Uses PyTorch kernels on CPU for near-native speed (**1.09x slowdown**) while maintaining SSD-streaming safety.
- **Universal Scale**: If it fits on your disk, it runs in VNN. Przetworzyliśmy **37GB** z prędkością **423 MB/s** na systemie z ograniczonym RAM.
- **100% Parity**: Verified against PyTorch for all core operations across all dtypes (int8 to int64).

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
| 📖 **[Technical Manual](docs/technical_manual.md)** | **Line-by-line source walkthrough** with detailed logic explanation. |

---

## 📁 Repository Structure

-   **[vulkan_nn_lib/](vulkan_nn_lib/)**: Core library (Shim, Kernels, ARAS engine).
-   **[docs/](docs/)**: Technical deep-dives and implementation details.
-   **[demos/splat_studio/](demos/splat_studio/)**: 3D Gaussian Splatting optimized with VNN.
-   **[tests/](tests/)**: Verification suite including **SSD Autograd 1GB tests**.

---

## 💎 Features

-   **SSD-Native Autograd**: Perform backpropagation on models that exceed RAM capacity.
-   **DRAS v4 (Adaptive RAM-Aware Streaming)**: Features **Adaptive Restart** and **Safety Violation Protection**.
-   **PyTorch Speed Parity**: RAM-resident CPU operations run at **~91% of PyTorch speed** (1.09x slowdown) and MatMul at **~73%** (1.36x slowdown).
-   **Zero-Copy Mounting**: Initialize tensors from binary files on disk in milliseconds without reading data.
-   **Full Dtype Support**: Verified support for `int64`, `int32`, `int16`, `int8`, and `float32`.

---
*Developed with 💙 for the open hardware community.*
