# 🗺️ VNN Roadmap: The Path to Universal AI

VNN (VulkanNN) has achieved 100% PyTorch parity and established an unparalleled foundation for extreme low-hardware training and inference. The project has successfully deployed the **Dual-Engine Architecture**, with the `vulkannn_rusted` engine now leading the charge for raw metal speed.

### ✅ Completed Milestones
- **Multi-Tiered Autograd**: Unified gradient flow across RAM, Vulkan, and SSD (Legacy Engine).
- **Native Extension (Rusted Ed)**: Deployed PyO3 and WGPU to bypass Python/Taichi bottlenecks.
- **Async 3-Stage Pipeline (v2.8)**: Triple-buffering system in `backend.rs` that overlaps I/O with compute.
- **CPU Superiority (v2.8)**: Achieved lower latency than PyTorch CPU in core MatMul/ReLU operations via Rayon and `matrixmultiply`.
- **Gemma 2B & 3 4B Support**: Verified engine performance on state-of-the-art weights.

### 🛠️ In-Progress: The "Commodity Survival" Strategy
Current development is focused on pushing the limits of native Rust implementation:

1. **Gemma 3n (MatFormer) Support (Short-Term)**
   - Implement `Tensor::slice()` and `Tensor::view()` to support "Elastic Inference".
   - Allow activating nested sub-models (E2B inside E4B) without re-allocating memory.
2. **PagedAttention inside WGPU (Rust Porting)**
   - Port the legacy Python PagedAttention block structures and logical virtualization into the Rust WGPU Buffer layouts.
3. **GGUF Unification (mmap Integration)**
   - Full support for GGUF headers and block-structured memory mapping in Rust.
4. **Asymmetric Speculative Decoding**
   - Implement a 2-engine pipeline where a tiny "Draft" model validates tokens locally in VRAM while the "Primary" weights stream from SSD.

### 🚀 Long-Term (Expansion)
1. **Multi-Precision Core (FP16/INT8/INT4)**
   - Implement FP16/BF16 kernels for 2x speedup on compatible hardware.
   - Native INT8/INT4 quantization with on-the-fly decompression inside WGSL shader registers (Ghost Quantization).
   - Aim for 4-bit support similar to PyTorch's `bitsandbytes` (NF4) but optimized for SSD streaming.
2. **Distributed VNN**
   - Parallelization across networked systems via gRPC tunnels built into `vulkannn_rusted`.
3. **ONNX/GGUF Native Support**
   - Universal model translation into VNN native format.

## 🛠️ The "Low-Hardware" Dream
Our ultimate goal is to enable **Llama-3 (70B) inference on systems with 4-8GB RAM & GPU** at interactive latencies by combining SSD DMA saturation with speculative draft validation.

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
