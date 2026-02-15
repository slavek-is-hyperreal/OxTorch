# 🌌 Tensor Forever (VulkanNN)

**Tensor Forever** is a universal, lightweight neural network engine built on **Taichi Vulkan**. It enables high-performance AI inference on virtually any GPU—from modern high-end cards to legacy hardware like AMD GCN 2—by bypassing CUDA and leveraging cross-platform SPIR-V kernels.

## 🚀 Key Features
- **Virtual VRAM (Weight Paging)**: Run models larger than your GPU's physical memory by streaming weights in real-time.
- **PyTorch Compatibility**: A seamless shim (`torch_shim.py`) allows you to run existing PyTorch-style code on Vulkan with zero modifications.
- **Vendor Agnostic**: Works on Linux/Windows across AMD, Intel, and NVIDIA GPUs.
- **Pure Python/Taichi**: Easy to modify, hack, and integrate without complex C++ build chains.

---

## 🏗 Project Structure

### 🛠 [Core Library (vulkan_nn_lib)](vulkan_nn_lib/)
The heart of the engine. Contains the `Tensor` abstraction, GPU kernels (Attention, RoPE, MatMul), and the PyTorch compatibility layer.

### 🎨 [Demo: Splat Studio](demos/splat_studio/)
**Graphics Demonstrator**.
A unified GUI for Gaussian Splatting and 3D reconstruction. It uses Tensor Forever to accelerate depth refinement and point cloud processing.

### 🗨 [Demo: Gemma Chat](demos/gemma_chat/)
**LLM Demonstrator**.
An interactive chat interface for the **Gemma 3 Nano** model. Showcases the engine's ability to handle complex Transformers, KV-Caching, and Matryoshka (MatFormer) weight slicing.

---

## ⚙️ Quick Start

### 1. Requirements
- Python 3.10+
- Vulkan-capable drivers (e.g., Mesa/RADV on Linux)
- A working Python Virtual Environment

### 2. Install Dependencies
```bash
pip install taichi numpy
```

### 3. Usage Example (PyTorch Hijacking)
```python
import vulkan_nn_lib.torch_shim as torch

# This now uses the Vulkan GPU backend!
x = torch.randn(1, 128)
linear = torch.nn.Linear(128, 64)
output = linear(x)
print(output.to_numpy())
```

---

## 📜 Documentation
For detailed API references, kernel specifications, and advanced usage, see the [Library README](vulkan_nn_lib/README.md).

---
*Created with 💙 for legacy hardware and the open AI ecosystem.*
