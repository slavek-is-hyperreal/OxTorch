# 🌌 VNN Legacy Edition (VulkanNN)

**"1:1 PyTorch Replacement for Aging Hardware"**

VNN is a high-performance tensor library designed to run massive AI models (Gemma, Llama) on older hardware with limited RAM but plenty of SSD storage. It leverages a **Streaming Operator Engine (SOE)** to process tensors far exceeding system memory.

---

## 📁 Repository Structure

-   **[vulkan_nn_lib/](vulkan_nn_lib/)**: Core library. A 1:1 replacement for `import torch`.
-   **[vulkan_nn_lib/](vulkan_nn_lib/)**: Core library. A 1:1 replacement for `import torch`.
-   **[demos/splat_studio/](demos/splat_studio/)**: 3D Gaussian Splatting and depth refinement demo (Optimized with VNN).
-   **[tests/](tests/)**: Comprehensive verification suite.

---

## 🚀 Quick Start (Verification)

1.  **Setup Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install taichi numpy tqdm
    ```

2.  **Run ARAS Benchmark**:
    ```bash
    PYTHONPATH=. python3 tests/test_phase3_aras.py
    ```

---

## 💎 Features

-   **SSD-Native Tensors**: Operations (Matrix Mult, Add, etc.) stream directly from disk to GPU.
-   **Auto-Budgeting**: Automatically moves tensors between RAM, VRAM, and SSD based on your hardware profile.
-   **Vulkan Accelerated**: Powered by Taichi-Vulkan for cross-vendor GPU support (Intel/AMD/NVIDIA).

---

## 🎨 Creative Demos

### Splat Studio (3D Graphics)
Optimized depth estimation running on the VNN core.
```bash
python3 demos/splat_studio/splat_studio.py
```

---
*Developed with 💙 for the open hardware community.*
