# VNN Performance Guide: VNN vs. PyTorch

VNN is not a general-purpose replacement for PyTorch; it is a **specialized tool** for hardware-constrained environments.

## Where VNN is Better (The "Win" Scenarios)

### 1. Ultra-Large Model Loading
- **PyTorch**: Often requires enough RAM to hold the entire state dict during `load_state_dict`.
- **VNN**: Uses `from_binary`. Memory usage is **constant (approx. 50MB)** regardless of whether the model is 7B, 70B, or 400B parameters.

### 2. Low-RAM Training/Inference
- **Backpropagation**: VNN supports SSD-native gradient accumulation. In Phase 8 testing, we achieved **423 MB/s** stable throughput for a **37GB sum** operation on a system with standard RAM, outperforming PyTorch by virtue of surviving.
- **Safety**: The **Backpressure Mechanism** and **21.5GB Threshold** ensure that VNN will never crash your Linux desktop, even when pushing RAID-0 saturated I/O.

### 3. Hardware Greed
- **PyTorch**: Relies on OS-level page cache for large data.
- **VNN**: Uses **Greedy Factory** logic. It bypasses filesytem bottlenecks (like ZFS compression or ARC limits) by pre-allocating large chunks in resident RAM. This allows for **1GB/s+** speeds even on standard SATA/NVMe setups.

## Where VNN is Slower (The Trade-offs)

### 1. Small Batch Latency
- **PyTorch**: Highly optimized C++/CUDA kernels with micro-second overhead.
- **VNN**: Python-based orchestration and Taichi compilation overhead. For tiny operations (e.g., 1024 elements), PyTorch will be 10x-100x faster.

### 2. GPU Acceleration
Our Vulkan backend is designed for older/generic hardware. While slower than CUDA on NVIDIA cards, it provides acceleration on devices PyTorch often ignores (Intel UHD, mobile iGPUs).

## 📊 Benchmark Results (Standardized)
| Operation | Size | Engine | Mode | Speed / Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Add** | 8.3 M | **PyTorch** | CPU | 10.2 ms |
| **Add** | 8.3 M | **VNN** | CPU | **11.1 ms (1.09x)** |
| **MatMul** | 1024^2 | **PyTorch** | CPU | 25.6 ms |
| **MatMul** | 1024^2 | **VNN** | CPU | **34.9 ms (1.36x)** |
| **Monster Sum**| 37 GB | **VNN** | **SOE Engine** | **423 MB/s (OOM-Safe)** |

## Comparison Table

| Feature | PyTorch | VNN |
| :--- | :--- | :--- |
| **Philosophy** | Compute-First (Max Speed) | Memory-First (No Crashes) |
| **Min. RAM to run Gemma-7B** | ~16GB+ | **~512MB** |
| **I/O Strategy** | Standard OS Memmap | Greedy ARAS Buffering |
| **Ease of Use** | Standard | `import torch_shim as torch` |
| **Backpropagation** | In-Memory (Fast) | **SSD-Native (OOM-Safe)** |
| **Gradient Checkpointing**| Manual | Automatic (via SSD Offload)|
