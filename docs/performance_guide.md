# ⚡ Performance & Stability Guide (v2.9.0)

This guide explains how to interpret benchmarks and maximize the throughput of VNN Rusted.

---

## 1. The Ratio Metric (VNN/PyTorch)
Because operating systems have background processes (Antigravity, browser, system updates), absolute execution time (s) is a "dirty" metric.
*   **VNN Ratio < 1.0**: VNN is faster than PyTorch.
*   **VNN Ratio > 1.0**: PyTorch is faster.
*   **Stability**: If both VNN and PyTorch slow down together, the Ratio remains stable. This is our primary way to track algorithmic efficiency.

---

## 2. Performance Stability Law
Observed in v2.9.0 benchmarks:
1.  **Ultra-Short Tasks (<5ms)**: High Coefficient of Variation (**CV ~40%**). These are dominated by Python overhead and OS context switching.
2.  **Medium Tasks (~100ms)**: **CV ~5-10%**. Dominated by GPU driver initialization and command submission.
3.  **Heavy Tasks (>10s)**: **CV < 1%**. These represent true algorithmic potential. At this scale, system noise is negligible.

---

## 3. Tiling Strategies
VNN Rusted uses adaptive tiling to fit within consumer VRAM (e.g., 2GB).
*   **GEMV Path (N=1)**: Optimized in `src/backend.rs:263` with forced N-tiling to enable Double Buffering.
*   **Large MatMul**: Uses **512x16384** A-tiles (`src/backend.rs:265`) to minimize cache misses.

---

## 4. Maximizing Throughput
1.  **Use `device="hybrid"`**: This uses all 4 cores of your i5-3450 AND the R7 200 GPU simultaneously.
2.  **SSD Mapping**: Use `Tensor.from_ssd`. VNN uses background threads (`src/streaming.rs:96`) to "touch" memory pages, ensuring the data is in RAM *before* the computation starts.
3.  **Warm-up**: Always run one small MatMul before benchmarking to let the Vulkan driver compile the WGSL pipelines.

---

## 5. Hardware Limitations & Throttling
*   **Power Limit**: If the GPU hits its power limit, you will see a sudden rise in **CV%**.
*   **Thermal Throttling**: Long runs (e.g., 16GB SSD tests) heat up the CPU/GPU, which can increase absolute times by up to 25% while the Ratio remains stable.
