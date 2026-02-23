# 🗺️ VNN Roadmap: The Path to Universal AI

VNN Legacy Edition has achieved 100% PyTorch parity and verified performance. This roadmap outlines the strategic direction for Phase 8 and beyond, shifting focus heavily toward the extreme constraints of legacy commodity hardware, exploiting memory suballocation, unified compression, and asymmetric pipelines.

### ✅ Completed Milestones
- **Multi-Tiered Autograd**: Unified gradient flow across RAM, Vulkan, and SSD.
- **DRAS v4**: Adaptive memory-aware streaming with backpressure.
- **Kaggle Mode**: Production-ready remote execution for massive tensors.
- **PyTorch Parity**: Verified numerical consistency and API compatibility.

### 🛠️ Hardware-First Paradigms: The "Commodity Survival" Strategy
Based on recent architectural research into extreme quantization and memory optimization for legacy platforms, the focus of VNN development is now prioritized into the following 5 strategic pillars:

1. **Suballocation Architecture (Memory Foundation)**
   - Eliminate direct buffer allocations (`vkAllocateMemory`) which exhaust application driver limits.
   - Implement Slab/Buddy suballocation (`VulkanTensorPool`) directly inside `memory.py` to allow thousands of virtual tensors using only a few massive physical allocations.
2. **PagedAttention & Context Management**
   - Address the staggering 80% VRAM waste of contiguous KV Caches.
   - Implement `BlockTable` based PagedAttention to map non-contiguous fragments memory seamlessly to physical VRAM blocks, achieving near-zero internal fragmentation on older cards caching massive contexts.
3. **Shader Physics and Micro-Architecture**
   - **Bank Conflict Elimination**: Rewrite Taichi matrix multiplication (`k_matmul`) using structural padding (e.g., modifying stride by +1) to dismantle serialized 4-way bank conflicts on Shared Memory.
   - **GLSL FlashAttention**: Compute the squared scaling and softmax aggregations entirely within shader registers to avoid expensive round-trips to the global memory interconnect.
4. **Asymmetric Pipelining & SpecOffload**
   - Traditional SSD offloading causes ALU idleness due to PCIe latency.
   - Implement **Speculative Decoding Pipelines**: Run a tiny "Draft Model" locally embedded inside VRAM concurrently while the primary model's massive weights transfer via SSD DMA. This turns PCIe waiting times into predictive validation tasks padding token generation.
5. **GGUF Unification and Sub-Byte Quantization**
   - Retire raw `.bin` SSD arrays in favor of block-structured **GGUF `mmap`** support.
   - Deploy Blockwise Quantizations (e.g., `Q4_K_M`) and test experimental TC-FPx 6-bit registers where scale vectors retain FP16 for precision, mitigating the severe accuracy drop of global int4 packing.

### 🚀 Mid-Term (Compression & Throughput)
- **Native Blockwise INT4/6 Decompression**: High-speed weight decompression inside Taichi shaders.
- **Prefetcher 2.0 (Speculative)**: Adaptive windowing for NVMe saturation interlaid with SpecOffload task delegation.

### 🌌 Long-Term (Expansion)
- [ ] **Distributed VNN**: Parallelization across networked systems via shared NVMe targets.
- [ ] **ONNX Integration**: Direct translation of models into our GGUF/VNN engine.

## 🛠️ The "Low-Hardware" Dream
Our ultimate goal is to enable **Llama-3 (70B) inference on systems with 4-8GB RAM & GPU** at interactive latencies.
- **Strategy**: Synergizing the Speculative Asymmetric pipeline, NVMe GGUF Saturation, and Vectorized FlashAttention.

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
