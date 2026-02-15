# 🌌 Tensor Forever (VulkanNN)

**Tensor Forever** is a universal neural engine that brings modern AI (LLMs) to hardware that was previously "too old" or "too weak". 

By using **Taichi Vulkan**, we bypass the need for expensive NVIDIA GPUs and CUDA. If your computer has a graphics card (even an old one from 2013 like an AMD R7 260X), **Tensor Forever** can "reanimate" it to run Large Language Models like Gemma 3.

---

## 🚀 Beginner's Guide: Start Chatting in 5 Minutes

If you just want to talk to an AI on your legacy hardware, follow these steps:

### 1. Requirements
You need **Linux** (e.g., Ubuntu, Mint) and **Python 3.10+**.
Open your terminal and run:
```bash
sudo apt update && sudo apt install python3-venv python3-tk
pip install taichi numpy transformers huggingface_hub tqdm safetensors
```

### 2. Setup Kaggle (For Model Weights)
We use weights from Google's Gemma 3. To download them automatically:
1. Go to [Kaggle.com](https://www.kaggle.com), log in, and click on your profile picture -> **Settings**.
2. Click **Create New Token** to download a `kaggle.json` file.
3. Create a folder named `vulkan_nn_lib` in this project and put `kaggle.json` inside it.

### 3. Run the Chat
Simply run this in your terminal:
```bash
# This will download the model, convert it, and start the chat!
python3 demos/gemma_chat/chat_gemma.py
```
*Note: The first run will take some time to download and convert the weights (approx. 8-16GB).*

---

## 🏗 Project Structure

### 🛠 [Core Library (vulkan_nn_lib)](vulkan_nn_lib/)
The heart of the engine. A standalone library you can import into your own Python projects to get GPU acceleration on any Vulkan card.

### 🎨 [Demo: Splat Studio](demos/splat_studio/)
**Graphics Demonstrator**.
Turn videos into 3D scenes (Gaussian Splatting) using Vulkan-accelerated depth processing.

### 🗨 [Demo: Gemma Chat](demos/gemma_chat/)
**LLM Demonstrator**.
The interactive chat interface for **Gemma 3 Nano**. It uses our unique "Weight Paging" to run a 4B parameter model on cards with very low VRAM (e.g., 2GB).

---

## 💡 Why 'Tensor Forever'?
Modern AI often requires buying new hardware every 2 years. We believe in sustainability and accessibility. **Tensor Forever** aims to make AI run on the hardware you already own, for as long as possible.

---
*Created with 💙 for legacy hardware and the open AI ecosystem.*
