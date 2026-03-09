# ⚡ Performance & Stability Guide (v3.2.0 "Valkyrie")

This guide explains how to interpret the Valkyrie Statistical Audit and maximize the throughput of VNN Rusted.

---

## 1. The Statistical Guard
Starting with v3.2.0, VNN uses multiple benchmark runs (`--runs N`) to filter out system noise (e.g., Anti-Gravity activity, browser tab rendering).

1.  **PT/VNN (Median)**: The primary metric. The median is robust against "spikes" caused by background OS processes.
2.  **StdDev (Standard Deviation)**: A measure of stability.
    - **< 5ms**: Excellent thermal stability.
    - **> 50ms**: Likely indicates **Thermal Throttling** or high background CPU contention.
3.  **Ratio (VNN/PT)**: The hardware-invariant speed factor.
    - **Ratio < 1.0**: VNN is faster.
    - **Ratio 0.001x**: Characteristic of VNN's massive advantage in F16/BF16 on hardware where PyTorch lacks native acceleration.

---

## 2. Hardware Thermal Law
Observed during the 10-run (9790s) stress test:
- **Phase 1 (Cold Start)**: Tensors are cold, cache is empty. Median times are consistent.
- **Phase 10 (Heat Soak)**: After ~2 hours of computation, absolute times may increase by **10-15%** due to CPU/GPU clock-down. However, the **Ratio** remains stable as both VNN and PyTorch are slowed equally by the hardware.

---

## 3. Tiling & Precision
VNN Rusted uses adaptive tiling to maximize bandwidth on mid-range GPUs (e.g., 2GB VRAM).
*   **MatMul Tiling**: `src/backend.rs:360` uses **512x16384** A-tiles. Large tiles reduce the number of command submissions but require more VRAM.
*   **Tri-Precision Fallback**: If using a legacy GPU (pre-2018), `backend.rs:214` will cast F16 data to F32 for the compute shader. You retain the **50% smaller SSD storage** of F16 but compute at F32 precision.

---

## 4. Maximizing Throughput (Hardware Saturation Method)
1.  **Close the Browser**: While VNN is resilient, heavy browser activity increases `StdDev` on CPU tasks.
2.  **Use `device="hybrid"`**: This saturates all CPU cores via Rayon (`src/tensor.rs:631`) while the GPU handles heavy SGEMM tiles via `wgpu` (`src/backend.rs:429`).
3.  **SSD Direct Streaming**: Always use `Tensor.from_ssd`. This uses the Linux kernel's DMA prefetching (`src/tensor.rs:92`) to stream weights directly into the computation pipeline, effectively turning your SSD into an L3 Cache.

---

## 5. Known Limitations
- **Small Kernels**: Operations like `ReLU` on small vectors (<2M elements) are faster on CPU because the Vulkan command submission overhead (~1ms) outweighs the compute time. VNN automatically handles this threshold in `src/backend.rs:597`.
