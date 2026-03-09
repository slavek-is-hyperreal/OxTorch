# 🗺️ VNN Roadmap: The Path to Universal AI

VNN (VulkanNN) has achieved 100% PyTorch parity and established an unparalleled foundation for extreme low-hardware training and inference. The project has successfully deployed the **Tri-Precision Engine (v3.2.0)**, supporting F32, F16, and BF16 across all backends.

### ✅ Completed Milestones
- **Multi-Precision Core (v3.2.0)**: Native `f16` and `bf16` support with CPU fast-paths and hardware-fallback GPU compute.
- **Statistical Audit (v3.2.0)**: Integrated Median/StdDev tracking to filter OS noise and monitor thermal patterns.
- **Async 3-Stage Pipeline (v2.8)**: Triple-buffering system in `backend.rs` that overlaps I/O with compute.
- **CPU Superiority**: Achieved lower latency than PyTorch CPU in core operations via Rayon and `matrixmultiply`.
- **SSD L3 Cache**: Memory mapping for tensors larger than system RAM (Verified up to 40k x 40k).

### 🛠️ In-Progress: The "Iron Age" Stability
Current development is focused on pushing the limits of the native Rust implementation:

1.  **Overnight Stress Testing (5000+ Iterations)**
    - Long-form validation to verify SSD aging and thermal clock-down behavior.
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

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
