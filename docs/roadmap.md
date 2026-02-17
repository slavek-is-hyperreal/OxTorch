# 🗺️ VNN Roadmap: The Path to Universal AI

VNN Legacy Edition has achieved 100% PyTorch parity and verified performance. This roadmap outlines the strategic direction for Phase 8 and beyond, focusing on compression, efficiency, and usability.

## 🎯 Short-Term (Stability & Refinement)
- [ ] **Lazy Buffer Allocation**: Delay SSD file creation until the first write operation to optimize disk space for intermediate computations.
- [ ] **Kernel Fusions**: Implement fused kernels (e.g., `Linear + ReLU + Add`) in Taichi to minimize GPU-CPU synchronization overhead.
- [ ] **Deterministic Testing**: Expand the parity suite to include deterministic seed testing for stochastic operations such as `randn` and `dropout`.

## 🚀 Mid-Term (Compression & Throughput)
- [ ] **Native INT4 Quantization**:
    - Develop specialized Taichi kernels for 4-bit weights with high-speed, on-the-fly decompression.
    - Target a 2x reduction in SSD storage requirements and a 1.5x throughput increase for "Monster Scale" models.
- [ ] **Prefetcher 2.0**: 
    - Implement a dynamic prefetch window that automatically adapts to SSD latency and bandwidth profiles (SATA vs. NVMe).
    - Introduce native support for multi-SSD striping (RAID-0 mode) directly within the library.
- [ ] **Advanced Fusion Logic**: Group multiple element-wise operations into a single SSD stream pass to eliminate redundant I/O cycles.

## 🌌 Long-Term (Ecosystem & Deployment)
- [ ] **Distributed VNN**: Support for multi-system training where SSD storage is shared over a 10GbE network.
- [ ] **ONNX Integration**: Tooling to import ONNX models directly into VNN SSD format.
- [ ] **Web Integration**: Explore WebGPU (via Taichi-JS) to bring VNN's SSD-streaming capabilities to high-end browser applications.

## 🛠️ The "Low-Hardware" Dream
Our ultimate goal is to enable **Llama-3 (70B) inference on systems with 8GB RAM** at interactive latencies (1-2 tokens/sec).
- **Strategy**: Synergizing aggressive INT4 quantization, NVMe saturation, and Fused Streaming Kernels.

---
*VNN is not just a library; it's a statement that AI should be accessible to everyone, regardless of their budget.*
