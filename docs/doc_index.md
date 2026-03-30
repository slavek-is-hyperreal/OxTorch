# Documentation Index (v3.8.1-rc)

Welcome to the OxTorch Documentation Index. This guide serves as a "GPS" to help you navigate the various technical documents, guides, and specifications of the project.

---

## 1. Core Architecture & Design
- **[Architecture](architecture.md)**: A high-level deep dive into OxTorch's components, including the MSTS v2 engine and unified pipeline.
- **[MSTS Logic](msts_logic.md)**: Theoretical foundation of the Multi-Stage Tensor Streaming system, bitmask barriers, and handshakes.
- **[SSD Storage](ssd_storage.md)**: Specification of the binary raw format, `O_DIRECT` requirements, and SSD-Direct dispatch.
- **[Tensor Pool](tensor_pool.md)**: Technical breakdown of the Deterministic Ring Buffer used for zero-copy memory management.

## 2. Backend & Performance
- **[CPU Backend](cpu_backend.md)**: Guide to SIMD specialized kernels (AVX/NEON) and S.O.P. for adding new functions.
- **[Kernel Report](kernel_report.md)**: Detailed optimization status and dispatch tiers for every operation.
- **[Performance Guide](performance_guide.md)**: Tips for maximizing throughput and understanding the 400x gain on legacy hardware.
- **[Support Matrix](support_matrix.md)**: Table of supported SIMD features and precisions across architectures.

## 3. Python API & Integration
- **[API Reference](api_reference.md)**: Comprehensive list of available Python methods and classes in the `oxtorch` proxy.
- **[OxTorch Package](oxtorch_package.md)**: Overview of the Python wrapper, dynamic dispatch, and fallback mechanisms.
- **[PyTorch Gap Analysis](pytorch_gap_analysis.md)**: Ongoing comparison of OxTorch features vs. PyTorch standard API.

## 4. Developer Resources
- **[How We Test](how_we_test.md)**: Documentation of the atomized benchmark suite and numerical parity protocols.
- **[Execution Modes](execution_modes.md)**: Characteristics of CPU, Vulkan GPU, Hybrid, and SSD Streaming modes.
- **[New Op Tutorial](new_op_tutorial.md)**: Step-by-step guide for developers adding new operations from Rust to Python.

---
*Last updated: 2026-03-30 (v3.8.1-rc)*
