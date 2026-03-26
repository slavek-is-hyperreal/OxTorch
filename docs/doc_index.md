# Documentation Index (The GPS)

Welcome to the OxTorch Documentation Index. This guide serves as a "GPS" to help you navigate the various technical documents, guides, and specifications of the project.

## Core Architecture & Design
- **[Architecture](architecture.md)**: A high-level deep dive into OxTorch's components, including the MSTS engine and backend layers.
- **[Execution Modes](execution_modes.md)**: Detailed comparison of CPU, Vulkan GPU, Hybrid, and SSD Streaming modes.
- **[MSTS Logic](msts_logic.md)**: Theoretical foundation of the Multi-Stage Tensor Streaming system and its state machine.
- **[MSTS Visualization](msts_visualization.md)**: Diagrams and state transitions of the MSTS CrookScheduler.
- **[Tensor Pool](tensor_pool.md)**: Technical breakdown of the slab allocator used for ultra-fast RAM management.

## Backend Deep Dives
- **[CPU Backend](cpu_backend.md)**: Guide to folder structure, SIMD kernels (AVX/NEON), and implementation rules for CPU ops.
- **[Vulkan Internals](vulkan_internals.md)**: Low-level details of the Ash (Vulkan 1.2) backend, memory pooling, and async execution.
- **[Support Matrix](support_matrix.md)**: A detailed table of supported SIMD features and precisions (F32, F16, BF16, Int8) across architectures.

## Storage & Distribution
- **[SSD Format](ssd_format.md)**: Specification of the binary `.ssd` format and direct I/O requirements.
- **[Binary Distribution](binary_distribution.md)**: Strategy for hardware-optimized binary releases and offline deployment.
- **[OxTorch Package](oxtorch_package.md)**: Overview of the Python wrapper, dynamic dispatch, and fallback mechanisms.

## Development & Testing
- **[New Op Tutorial](new_op_tutorial.md)**: Step-by-step guide for developers adding new operations from Rust to Python.
- **[Implementation Guides](implementation_guides.md)**: Templates and roadmaps for upcoming feature phases.
- **[How We Test](how_we_test.md)**: Documentation of the atomized benchmark suite and regression testing protocol.
- **[BitNet Status](bitnet_status.md)**: Current technical state, blockers, and functional parity report for BitNet-2B/3B.
- **[PyTorch Gap Analysis](pytorch_gap_analysis.md)**: Ongoing comparison of OxTorch features vs. PyTorch standard API.

## Project Management
- **[Roadmap](roadmap.md)**: Long-term vision and phase-by-phase development goals.
- **[Changelog](CHANGELOG.md)**: History of releases and significant technical improvements.
- **[Performance Guide](performance_guide.md)**: Tips for maximizing throughput on various hardware configurations.
- **[Notes](notes.md)**: Internal developer scratchpad for architectural revisions and future ideas.
- **[API Reference](api_reference.md)**: Comprehensive list of available Python methods and classes.

---
*Last updated: March 2026*
