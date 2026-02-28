# 🗺️ VNN Roadmap: The Path to Universal AI

VNN (VulkanNN) has achieved 100% PyTorch parity and established an unparalleled foundation for extreme low-hardware training and inference. The project has successfully deployed the **Dual-Engine Architecture**, with the `vulkannn_rusted` engine now leading the charge for raw metal speed.

### ✅ Completed Milestones
- **Multi-Tiered Autograd**: Unified gradient flow across RAM, Vulkan, and SSD.
- **Kaggle Mode**: Production-ready remote execution for massive tensors in the legacy engine.
- **Native Extension (Rusted Ed)**: Deployed PyO3 and WGPU to bypass Python/Taichi bottlenecks.
- **True Heterogeneous Compute**: Asynchronous Rayon (CPU) and WGPU staging buffer processing working concurrently in Rust.

### 🛠️ Hardware-First Paradigms: The "Commodity Survival" Strategy
Current and active development is focused heavily on exploiting the Rust bare-metal foundation:

1. **[x] Suballocation Architecture (Memory Foundation)** (COMPLETED)
   - Bypassed physical Vulkan driver allocation counts with a Slab/Buddy sub-allocator in `vulkan_nn_lib`.
2. **[x] Native C/Rust WGSL Pipeline** (COMPLETED)
   - Replaced JIT shaders with explicitly compiled WGSL compute shaders inside `vulkannn_rusted`.
3. **Asymmetric Pipelining & SpecOffload (Next Generation)**
   - Traditional SSD offloading causes ALU idleness due to PCIe latency.
   - Implement **Speculative Decoding Pipelines**: Run a tiny "Draft Model" locally embedded inside VRAM concurrently while the primary model's massive weights transfer via SSD DMA. This turns PCIe waiting times into predictive validation tasks padding token generation.
4. **PagedAttention inside WGPU (Rust Porting)**
   - Port the legacy Python PagedAttention block structures and logical virtualization into the Rust WGPU Buffer layouts.
5. **GGUF Unification and Sub-Byte Quantization**
   - Retire raw `.bin` SSD arrays in favor of block-structured **GGUF `mmap`** support natively inside Rust via `memmap2`.
   - Deploy Blockwise Quantizations (e.g., `Q4_K_M`) within WGSL registers.

### 🚀 Mid-Term (Compression & Throughput)
- **Native Blockwise INT4/6 Decompression**: High-speed weight decompression inside WGSL shader registers.
- **Speculative Prefetcher V2**: Adaptive sliding window DMA inside `streaming.rs` driven by `tokio`.

### 🌌 Long-Term (Expansion)
- [ ] **Distributed VNN**: Parallelization across networked systems via shared NVMe targets using gRPC tunnels built into `vulkannn_rusted`.
- [ ] **ONNX Integration**: Direct native translation of models into our GGUF/VNN engine.

## 🛠️ The "Low-Hardware" Dream
Our ultimate goal is to enable **Llama-3 (70B) inference on systems with 4-8GB RAM & GPU** at interactive latencies.
- **Strategy**: Synergizing the Speculative Asymmetric pipeline, NVMe GGUF Saturation, and Vectorized FlashAttention entirely in Rust.

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
