# 🗺️ VNN Roadmap: The Path to Universal AI

VNN (VulkanNN) has achieved 100% PyTorch parity and established an unparalleled foundation for extreme low-hardware training and inference. The project has successfully deployed the **Tri-Precision Engine (v3.2.0)**, supporting F32, F16, and BF16 across all backends.

### ✅ Completed Milestones
- **Iron Age CPU/SSD Optimizations (v3.3.0)**: Integrated `io_uring` direct I/O streaming mapped exclusively to MERA-400 inspired MSTS circular buffers. Eliminated CPU bottleneck via native branchless AVX1/SSE2 SWAR bit-twiddling for F16/BF16 types, reaching 600x PyTorch parity speed.
- **Multi-Precision Core (v3.2.0)**: Native `f16` and `bf16` support with CPU fast-paths and hardware-fallback GPU compute.
- **Statistical Audit (v3.2.0)**: Integrated Median/StdDev tracking to filter OS noise and monitor thermal patterns.
- **Async 3-Stage Pipeline (v2.8)**: Triple-buffering system in `backend.rs` that overlaps I/O with compute.
- **CPU Superiority**: Achieved lower latency than PyTorch CPU in core operations via Rayon and `matrixmultiply`.
- **SSD L3 Cache**: Memory mapping for tensors larger than system RAM (Verified up to 40k x 40k).
- **Raw `ash` Vulkan Backend (v3.3.0)**: Completely rewritten GPU backend from `wgpu` to `ash`. Decoupled GPU compute and PCIe DMA transfers using explicit Vulkan `Transfer` and `Compute` queues synchronized natively by Timeline Semaphores. Enabled dynamic FP16 support detecting hardware limits (e.g. AMD Bonaire).

### 🛠️ In-Progress: The "Iron Age" Stability
Current development is focused on pushing the limits of the native Rust implementation:

1.  **Long-Term Stress Testing (5000+ Iterations)**
    - Long-term test validation to verify SSD aging and thermal clock-down behavior.
2.  **Gemma 3n (MatFormer) Support**
    - Implement `Tensor::slice()` and `Tensor::view()` to support "Elastic Inference".
3.  **PagedAttention inside WGPU (Rust Porting)**
    - Port legacy Python PagedAttention logic into optimized WGSL storage layouts.
4.  **GGUF Unification**
    - Full support for GGUF headers and block-structured memory mapping.

### 🚀 Long-Term (Expansion)
1.  **Native INT8/INT4 Quantization**
    - On-the-fly decompression inside WGSL shader registers (Ghost Quantization).
    - Aim for 4-bit support optimized for SSD streaming.
2.  **Distributed VNN**
    - Parallelization across networked consumer laptops via gRPC tunnels.
3.  **Asymmetric Speculative Decoding**
    - 2-engine pipeline: tiny "Draft" model in VRAM validates tokens from a "Primary" model streaming from SSD.

### ⚙️ Phase 3: The "Iron Age" Extreme Optimizations
Inspired by the legendary Polish minicomputer [MERA-400](https://pl.wikipedia.org/wiki/Mera_400) (więcej informacji na [Mera400.pl](https://mera400.pl/Strona_g%C5%82%C3%B3wna)) and its innovative CROOK OS, this phase focuses on pushing the architectural limits of constrained hardware.
~~1.  **Out-of-Core `io_uring` Engine**~~ (Completed v3.3.0)
    - Direct O_DIRECT disk streaming acting as extreme software DMA, completely bypassing Linux VFS page caches to prevent memory thrashing on ZFS.
~~2.  **MERA Style Task Scheduler (MSTS)**~~ (Completed v3.3.0)
    - Lockless, Tagged-Token Dataflow circular buffer inspired by the MERA-400 CROOK system. Autonomous CPU/GPU workers polling atomic `StatefulTile` memory aligned strictly to 1MB ZFS boundaries.
~~3.  **Branchless AVX1 Bit-Twiddling (SWAR)**~~ (Completed v3.3.0)
    - Replacing slow scalar `f32::from()` iterators with raw `std::arch::x86_64` intrinsics, heavily simulating hardware F16C operations lacking in processors like i5-3450. Includes dynamic dispatch fallback to native F16C/AVX2 on newer architectures to maintain maximum performance.
~~4.  **Asynchronous Multi-Queue Vulkan Pipelining (Raw `ash` Rewrite)**~~ (Completed v3.3.0)
    - Completely rewriting the GPU backend from `wgpu` to `ash` (raw Vulkan). Decoupling GPU compute and PCIe DMA transfers using explicit Vulkan `Transfer` and `Compute` queues synchronized natively by Timeline Semaphores.

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
