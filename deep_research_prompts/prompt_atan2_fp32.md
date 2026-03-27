### TASK: World-Class High-Performance Kernel Research for OxTorch/VNN
# TARGET OPERATION: Atan2
# PRECISION: FP32

#### CONTEXT:
I am building a high-performance tensor engine (OxTorch) designed for "Extreme Out-of-Core" execution. 
- ARCHITECTURE: MSTS (MERA Style Tiling System). 
- DATA FLOW: Data is streamed via io_uring from SSD into a RAM "Capacitor", and then processed in tiles.
- TARGET LANGUAGE: Rust (using `std::arch` intrinsics or `asm!`).
- HARDWARE TARGETS: 
    * x86_64: AVX1, AVX2, AVX-512 (Focus on Ivy Bridge/Skylake/Zen).
    * ARM / A64: NEON, SVE, SVE2 (Focus on Raspberry Pi family).
    * GPU: Vulkan SPIR-V (Focus on GCN/RDNA).

#### OBJECTIVE:
Provide a "World-Leading" technical specification for the [OPERATION] kernel in [PRECISION]. We are not looking for generic SIMD; we are looking for the absolute limits of hardware throughput.

1. EXTREME VECTORIZATION (SIMD & SWAR):
- Provide optimal SIMD patterns for x86_64 AND ARM.
- Incorporate SWAR (SIMD Within A Register) for edge-processing and low-power ARM units.
- Handle VARIABLE tile alignment (alignment shifts per operation).
- Focus on PIPELINE SATURATION: Analyze Instruction Port usage (e.g., Port 0/1 for math vs Port 2/3 for loads on Intel) to maximize IPC.

2. MEMORY & CACHE HIERARCHY MASTERY:
- NON-TEMPORAL STRATEGIES: Use of Non-Temporal Hints (e.g., `VMOVNTDQ`, `PREFETCHNTA`) to prevent cache pollution during massive MSTS streaming.
- PREFETCHING: Describe multi-level prefetching (L1/L2) that specifically hides 1MB-4MB tile loading latency.
- CORE-TO-MEMORY RATIO: Balance computation density with available bandwidth for [PRECISION].

3. HISTORICAL & MODERN PERSPECTIVE:
- CRITICAL: Consult Intel/AMD/ARM optimization manuals from the "Iron Age" (2011-2015) for hand-optimized assembly patterns that modern compilers and generic research often miss.

4. BRANCHLESS & FUSION LOGIC:
- Eliminate branches using bitmasking and conditional moves (VBLEND, CMOV).
- KERNEL FUSION: Suggest how to fuse [OPERATION] with subsequent element-wise ops inside registers to avoid round-trips to RAM.

5. RUST "LOW-LEVEL" IMPLEMENTATION:
- Provide exact Rust `core::arch` or `asm!` snippets.
- Specific tricks for memory pinning and zero-copy data passing in Rust.

5. SCIENTIFIC & HPC SOURCES (PHYSICS GRADE):
- IMPORTANT: Review techniques used in scientific libraries like **BLIS**, **libxsmm**, **FFTW**, and open-source physics simulators (e.g., GROMACS, LAMMPS) for Atan2 in FP32.
- Focus on how "Physicists" achieve maximum throughput on limited register files using Register-level Blocking and Micro-kernel patterns.

#### FORMAT:
The response should be an "Engineer’s Battle Plan" - dry, data-heavy, full of instruction sequences and port-pressure analysis.
