---
description: Technical Documentation & External References for VulkanNN
---
# Technical References Policy

**For Future Agents and Developers:**
This project relies on high-performance, low-level Rust and Vulkan implementations. When researching core engine updates, consult the following high-quality external repositories which mirror VNN's technical philosophy.

## 1. Vulkan & Ash (Backend)
- **[vulkan-examples-rs](https://github.com/adrien-ben/vulkan-examples-rs)**: The primary reference for modern Rust bindings (`ash`). Focus on compute-only examples and timeline semaphores.
- **[unknownue/Vulkan](https://github.com/unknownue/Vulkan)**: Rust port of Sascha Willems' Vulkan samples. Excellent for troubleshooting specific backend features.

## 2. ML Engine Architecture (Rust)
- **[HuggingFace Candle](https://github.com/huggingface/candle)**: A minimalist ML framework in Rust. Use this to understand how to structure pure-Rust tensor dispatch and lazy evaluation patterns.

## 3. High-Performance I/O & CPU Optimization
- **[io-uring-example](https://github.com/tirr-c/io-uring-example)**: Reference for low-level asynchronous I/O. Use this for the "Sprint 3 - io_uring O_DIRECT" implementation.
- **[GGML (ggerganov/ggml)](https://github.com/ggerganov/ggml)**: The definitive source for GGUF format and CPU SIMD (AVX/ARM) optimization strategies on legacy hardware.

## 4. PyTorch Parity Research
- **[Papers-To-Code](https://github.com/Nikunjgoyal07/Papers-To-Code)**: Contains clean, readable implementations of transformer operations in PyTorch. Use this for verifying parity logic without digging through PyTorch's complex C++ core.

---
**Note:** Always prioritize repositories that favor explicit over implicit resource management, as OxTorch targets highly constrained (sub-2GB VRAM) environments.
