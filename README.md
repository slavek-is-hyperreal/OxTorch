# 🌌 VNN Legacy Edition (VulkanNN)

**"Because it SHOULD be possible to run AI on your old hardware."**

VNN is a hobbyist-born, research-focused tensor library designed to prove that massive AI models don't just belong to big tech clusters. It bridges the gap between 100GB+ models and the systems gathering dust in our home offices—PCs with limited RAM, older GPUs, and modest laptops.

---

### 📼 The Philosophy: Hardware-Defiant AI
Modern AI development is a race for raw FLOPS, often leaving anyone with less than 24GB of VRAM in the "Out of Memory" (OOM) zone. **VNN takes the alternative path.** 

We prioritize **Stability over Speed**. By treating your SSD as a first-class memory tier, VNN allows you to train and run models that physically shouldn't fit on your machine. If you have the disk space, you have a path to run the computation. No crashes, no compromises on scale.

---

### 🧪 Experimental Demo: Splat Studio
VNN powers **Splat Studio**, an end-to-end 3D Gaussian Splatting pipeline. 

> [!WARNING]
> **Status: Experimental & Untested.**
> This demo serves as a technical showcase for VNN's SSD-streaming `AutoAdam` optimizer. It is provided for research and curiosity; it has not been validated for production workloads.

- **Tiled Optimization**: Uses VNN's `AutoAdam` to train models with millions of points by offloading gradients directly to SSD.
- **Vulkan Rasterization**: High-speed visualization powered by the same Vulkan/Taichi backend as VNN core.
- **Experimental Suite**: Integrated tools for converting Gaussian Splats to meshes (`gs_to_mesh.py`) and extracting camera data.

To explore the demo:
```bash
PYTHONPATH=. python3 demos/splat_studio/splat_studio.py
```

---

## 🛠️ Core Capabilities

PyTorch is for the data center. **VNN is for the rest of us.**

-   **SSD-Native Autograd**: The jewel of VNN. Backpropagation that streams directly to/from disk, enabling training on models weighing hundreds of gigabytes.
-   **DRAS v4 (Adaptive RAM-Aware Streaming)**: Real-time memory monitoring that pushes your hardware to the absolute limit without crossing the RAM "cliff."
-   **Vulkan/Taichi Engine**: Hardware-agnostic compute that runs on Intel, AMD, and NVIDIA alike.
-   **100% Parity**: Mathematically verified against PyTorch for core operations.

> [!TIP]
> You can configure the SSD cache directory by setting the `VNN_CACHE_DIR` environment variable (defaults to `./vnn_cache`).

---

## 🚀 Drop-in Replacement (Shim)

VNN includes a PyTorch Shim, allowing you to refactor existing code with minimal effort:

```python
import vulkan_nn_lib.torch_shim as torch

# 1. Initialize a "Monster" Tensor (e.g., 10GB+)
# If size > RAM, VNN automatically mounts it on SSD.
w = torch.randn(1024, 1024, 2560, requires_grad=True) 

# 2. SSD-Native Math
loss = (w * 2.0).sum()

# 3. OOM-Safe Back propagation
loss.backward()

print(f"Gradient safely resident on: {w.grad.device}") # -> 'ssd'
```

---

## 📚 Technical Manuals

For those who want to see how the "magic" works:

| Guide | Description |
| :--- | :--- |
| 🏗️ **[Architecture](docs/architecture.md)** | Tiled memory logic, ARAS engine, and SSD backend. |
| 📜 **[Tensor API](docs/tensor_api.md)** | Deep dive into supported operations and dtypes. |
| ⚡ **[Performance Guide](docs/performance_guide.md)** | Real-world benchmarks (VNN vs Torch CPU). |
| 📖 **[The Manual](docs/technical_manual.md)** | **Line-by-line source walkthrough** explaining the implementation. |

---

## 💎 Features at a Glance

- **Memory-First Design**: Built to prevent OOM crashes at all costs.
- **Zero-Copy Loading**: Mount models from disk in milliseconds.
- **Full Dtype Support**: Verified for `int8` through `int64` and `float32`.
- **Hobbyist Friendly**: Optimized for 1GB - 4GB VRAM targets.

---
*Developed with 💙 for the open hardware and self-hosting community.*
