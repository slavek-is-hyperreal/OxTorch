# VNN Performance Guide: VNN vs. PyTorch

VNN is not a general-purpose replacement for PyTorch; it is a **specialized tool** for hardware-constrained environments.

## Where VNN is Better (The "Win" Scenarios)

### 1. Ultra-Large Model Loading
- **PyTorch**: Often requires enough RAM to hold the entire state dict during `load_state_dict`.
- **VNN**: Uses `from_binary`. Memory usage is **constant (approx. 50MB)** regardless of whether the model is 7B, 70B, or 400B parameters.

### 2. Low-RAM Training/Inference
- **PyTorch**: Crashes with `RuntimeError: CUDA out of memory` or `OOM Killed`.
- **VNN**: Automatically switches to **ARAS (SSD Streaming)**. It is slower than VRAM, but it **never crashes**. 
- **Backpropagation**: VNN now supports SSD-native gradient accumulation. In our "Monster Scale" tests, a 1GB parameter was trained with a backward pass latency of ~60s, maintaining stability where PyTorch would fail.

### 3. Hardware Greed
- **PyTorch**: Relies on OS-level page cache for large data.
- **VNN**: Uses **Greedy Factory** logic. It bypasses filesytem bottlenecks (like ZFS compression or ARC limits) by pre-allocating large chunks in resident RAM. This allows for **1GB/s+** speeds even on standard SATA/NVMe setups.

## Where VNN is Slower (The Trade-offs)

### 1. Small Batch Latency
- **PyTorch**: Highly optimized C++/CUDA kernels with micro-second overhead.
- **VNN**: Python-based orchestration and Taichi compilation overhead. For tiny operations (e.g., 1024 elements), PyTorch will be 10x-100x faster.

### 2. High-End GPU Scenarios
- **VNN**: Our Vulkan backend is currently optimized for stability and older hardware (Intel UHD, old Radeons).
- **PyTorch**: Native CUDA/ROCm will always beat a Vulkan abstraction on high-end NVIDIA/AMD cards.

## Comparison Table

| Feature | PyTorch | VNN |
| :--- | :--- | :--- |
| **Philosophy** | Compute-First (Max Speed) | Memory-First (No Crashes) |
| **Min. RAM to run Gemma-7B** | ~16GB+ | **~512MB** |
| **I/O Strategy** | Standard OS Memmap | Greedy ARAS Buffering |
| **Ease of Use** | Standard | `import torch_shim as torch` |
| **Backpropagation** | In-Memory (Fast) | **SSD-Native (OOM-Safe)** |
| **Gradient Checkpointing**| Manual | Automatic (via SSD Offload)|
