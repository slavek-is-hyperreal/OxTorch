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

---

## 🛠️ Core Capabilities

PyTorch is for the data center. **VNN is for the rest of us.**

-   **SSD-Native Autograd**: The jewel of VNN. Backpropagation that streams directly to/from disk, enabling training on models weighing hundreds of gigabytes.
-   **DRAS v4 (Adaptive RAM-Aware Streaming)**: Real-time memory monitoring with **Adaptive Backoff**. It pushes your hardware to the absolute limit without crossing the RAM "cliff," pausing I/O when the processing pipeline is saturated.
-   **Vulkan/Taichi Engine**: Hardware-agnostic compute that runs on Intel, AMD, and NVIDIA alike.
-   **Kaggle Offloading**: Zero-cost ephemeral supercomputing via Kaggle kernels.
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

## 🎛️ Device Selection

VNN gives you full control over where your tensors live. You can let the system manage it or force specific hardware.

```python
# 1. auto (Default)
# Intelligent placement based on RAM budget and tensor size.
# Small -> Vulkan/CPU, Large -> SSD.
x = Tensor(data, device='auto')

# 2. cpu
# Forces execution on System RAM. Good for small debugging or legacy ops.
x = Tensor(data, device='cpu')

# 3. vulkan
# Forces execution on GPU. 
# WARNING: Will crash if execution exceeds VRAM. Use for maximum speed on small-mid data.
x = Tensor(data, device='vulkan')

# 4. ssd
# Forces data to reside on disk (memory-mapped).
# Infinite capacity, limited by disk speed.
x = Tensor(data, device='ssd')
```

---

## ☁️ Kaggle Mode (Infinite Compute)

VNN can transparently offload massive computations to **Kaggle Kernels** (free T4 x2 GPUs), effectively giving you an ephemeral supercomputer.

### Setup
1. Get your `kaggle.json` API key from [Kaggle Settings](https://www.kaggle.com/settings).
2. Place it in the project root or `~/.kaggle/kaggle.json`.
3. Enable the mode via environment variable:

```bash
export VNN_KAGGLE_MODE=1
```

### How it works
- **Automatic Offloading**: When an operation (e.g., `MatMul`, `Add`) exceeds the `VNN_KAGGLE_THRESHOLD` (default: 1GB), VNN intercepts it.
- **Data Sync**: Inputs are uploaded as private Kaggle Datasets.
- **Remote Execution**: A specialized kernel is spun up to process the data on high-end GPUs.
- **Result Streaming**: Results are downloaded directly to your local SSD.
- **Seamless**: Your local Python script waits as if it were a local function call, transparently handling partitioning for tensors larger than Kaggle's 13GB RAM limit.

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

## 🌳 Repository Branching Strategy
- **`main`**: The public, stable version. Verified and ready for production/hobbyist use.
- **`test`**: Release candidate branch for comprehensive parity testing.
- **`dev`**: Active development of new experimental features. 

---
*Developed with 💙 for the open hardware and self-hosting community.*
