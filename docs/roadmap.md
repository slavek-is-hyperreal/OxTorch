# 🗺️ VNN Roadmap: The Path to Universal AI

VNN Legacy Edition has achieved 100% PyTorch parity and verified performance. This roadmap outlines the strategic direction for Phase 8 and beyond, focusing on compression, efficiency, and usability.

## 🎯 Short-Term (Stability & Refinement)
- [ ] **Lazy Buffer Allocation**: Delay SSD file creation until the first write to save disk space for intermediate operations.
- [ ] **Kernel Fusions**: Implement fused kernels (e.g., `Linear + ReLU + Add`) in Taichi to reduce GPU-CPU synchronization overhead.
- [ ] **Deterministic Testing**: Expand the parity suite to include deterministic seed testing for stochastic operations (`randn`, `dropout`).

## 🚀 Mid-Term (Compression & Throughput)
- [ ] **Native INT4 Quantization**:
    - Build specialized Taichi kernels for 4-bit weights with on-the-fly decompression.
    - Achieve a 2x reduction in SSD storage and 1.5x speedup for "Monster Scale" models.
- [ ] **Prefetcher 2.0**: 
    - Implement a dynamic prefetch window that adapts to SSD latency and SATA/NVMe bandwidth profiles.
    - Support for multiple SSDs (Data Stripping/RAID-0 mode) directly within the library.
- [ ] **Advanced Fusions**: Group multiple element-wise operations into a single SSD stream pass to eliminate redundant I/O.

## 🌌 Long-Term (Ecosystem & Deployment)
- [ ] **Distributed VNN**: Support for multi-system training where SSD storage is shared over a 10GbE network.
- [ ] **ONNX Integration**: Tooling to import ONNX models directly into VNN SSD format.
- [ ] **Web Integration**: Explore WebGPU (via Taichi-JS) to bring VNN's SSD-streaming capabilities to high-end browser applications.

## 🛠️ The "Low-Hardware" Dream
Our ultimate goal is to enable **Llama-3 (70B) inference on a laptop with 8GB RAM** with acceptable latency (1-2 tokens/sec).
- **Strategy**: Aggressive INT4 quantization + NVMe RAID saturation + Fused Streaming Kernels.

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
