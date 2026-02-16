# 🌌 Tensor Forever (VulkanNN)

**"Run Modern AI on Legacy Hardware"**

**Tensor Forever** is a magic engine that lets you run powerful AI models (like Google's Gemma 3) on old or weak computers. If you have *any* graphics card from the last ~10 years (even generic integrated Intel graphics), you can run this.

---

## 🚀 TL;DR: Quick Start Guide

**Goal:** Chat with Gemma 3 AI on your Linux computer in 5 minutes.

### 1. Requirements
-   **OS**: Linux (Ubuntu, Mint, etc.)
-   **Python**: Version 3.10 or newer.
-   **GPU**: Any card with Vulkan support (Intel HD 6xx, AMD R7/RX, NVIDIA GTX 6xx+).

### 2. Setup
Open your terminal and paste these commands:

```bash
# Install system tools
sudo apt update && sudo apt install python3-venv python3-tk

# Install python libraries
pip install taichi numpy transformers huggingface_hub tqdm safetensors
```

### 3. Get Model Access (One-time)
1.  Go to [Kaggle.com](https://www.kaggle.com), log in, and click your profile picture -> **Settings**.
2.  Scroll down to "API" and click **Create New Token**.
3.  A file named `kaggle.json` will download.
4.  Copy this file into a folder named `vulkan_nn_lib` inside this project directory.

### 4. Run the Chat!
```bash
python3 demos/gemma_chat/chat_gemma.py
```
*(The first time you run this, it will download the model weights automatically. This may take 10-20 minutes depending on your internet speed.)*

---

## ⚙️ Hardware Tuning (The "Sweet Spot")

Because **Tensor Forever** uses VRAM as a cache, you can tune it to your specific card:
-   **Low VRAM (2GB)**: Keep `tile_size` around 64MB to avoid stuttering.
-   **High RAM (32GB+)**: You can increase the system's "patience" to load huge models without GPU crashes.

Check the **[Calibration Guide](docs/ARCHITECTURE.md#7-hardware-calibration--tuning)** for pro-tips on VRAM BAR optimization.

---

## 🧠 For Developers & Geeks

Want to know how we fit a 4 Billion Parameter model into 2GB VRAM? Curious about our custom Autograd engine built on Vulkan compute shaders?

👉 **[Read the Deep Dive Architecture & Internals Guide](docs/ARCHITECTURE.md)**

There you will find details on:
-   **TiledLinear**: Our unique RAM-VRAM paging system.
-   **The `Tensor` Class**: How we wrap raw GPU memory.
-   **Autograd**: How we implemented backpropagation from scratch.

---

## 🎨 Other Demos

### Splat Studio (3D Graphics)
Turn videos into 3D scenes using Gaussian Splatting.
```bash
python3 demos/splat_studio/splat_studio.py
```

---
*Created with 💙 for the open AI ecosystem.*
