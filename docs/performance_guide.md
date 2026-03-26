# Performance Guide (v3.7.0 — The BitNet Leapfrog)

This guide explains how to interpret benchmark results and maximize throughput
on the reference hardware: i5-3450 (Ivy Bridge), AMD Radeon R7 200 (Bonaire GCN 1.1), 24GB DDR3, ZFS SSD pool.

---

## 1. Running Benchmarks

The primary benchmark system is the **Atomized Suite**:

```bash
source venv/bin/activate
PYTHONPATH=/path/to/vulkannn_rusted python tests/run_all_benchmarks.py
```

Output format (one line per test):
```
MatMul_f16_vulkan   | PT: 148.1644s | VNN:   0.1829s | 0.0012x | ✅ PASS
```

For full details see [how_we_test.md](how_we_test.md).

---

## 2. Benchmark Results (v3.7.0, AMD Bonaire / i5-3450)

### MatMul — OxTorch crushes PyTorch on legacy hardware

| Test | PyTorch | OxTorch | Ratio | Why? |
|:---|---:|---:|:---:|:---|
| MatMul F16 (vulkan) | 120.9s | 0.17s | **~0.0014x** 🚀 | ~780x faster via tiled Vulkan shader |
| MatMul BF16 (vulkan) | 68.9s | 0.17s | **~0.0025x** 🚀 | ~400x faster via SSE2+Vulkan |
| MatMul F16 (cpu) | 132.6s | 0.17s | **~0.0013x** 🚀 | F16C intrinsics vs PT scalar path |
| MatMul F32 (vulkan) | 0.22s | 0.18s | **0.82x** ✅ | Vulkan tiled SGEMM |
| MatMul INT8 (cpu) | 1.01s | 0.15s | **0.15x** 🚀 | 6.5x faster vs PT scalar |

> **Why PyTorch is so slow on F16/BF16 MatMul**: The i5-3450 has AVX+F16C but **not AVX-512**. PyTorch's CPU F16 MatMul requires AVX-512 — without it, it falls back to scalar emulation. OxTorch uses F16C intrinsics (`_mm256_cvtps_ph`) which are available on Ivy Bridge.

### Activations — Mixed results

| Test | PyTorch | OxTorch | Ratio | Notes |
|:---|---:|---:|:---:|:---|
| ReLU INT8 (cpu) | 3.1ms | 0.26ms | **0.085x** 🚀 | Dedicated INT8 SIMD kernel |
| ReLU F32 (cpu) | 4.0ms | 1.8ms | **0.44x** ✅ | AVX1 `vmaxps` |
| ReLU 15M F16 (hybrid) | 82.7ms | 50.8ms | **0.62x** ✅ | MSTS tile-pulling |
| ReLU 15M F32 (vulkan) | 24.3ms | 73.2ms | 3.01x ⚠️ | Vulkan PCIe overhead on 15M elems |
| ReLU 15M INT8 (hybrid) | 2.9ms | 46.3ms | 15.8x ⚠️ | PCIe cost kills small INT8 tensors |

---

## 3. Hardware Characteristics

### CPU (i5-3450, Ivy Bridge)

- 4 cores, no hyperthreading, 6MB L3 cache
- **Has**: AVX, F16C, SSE4.1 — **No**: AVX2, FMA, AVX-512
- F16/BF16 dispatch uses F16C intrinsics (F16) or SSE2 SWAR (BF16)
- F32 MatMul: near 1:1 with PyTorch (~0.21s for 2048×2048)
- F16/BF16 MatMul: **~500× faster** than PyTorch (scalar emulation vs F16C)

### GPU (AMD Radeon R7 200 Series, Bonaire GCN 1.1)

- ~1GB GDDR5 VRAM (effective compute pool via gpu-allocator)
- PCIe 3.0 — round-trip staging cost: ~80ms on Bonaire
- **No native FP16 compute in SPIR-V**: all shaders operate on F32; F16/BF16 is converted on CPU before upload
- GPU break-even: **≥4M elements** (~16MB F32, ~8MB F16)
- Below threshold: Vulkan overhead makes CPU faster

### SSD (ZFS pool)

- Effective sequential read: ~80–90 MB/s for Monster benchmarks
- `io_uring` + `O_DIRECT` + 1MB recordsize alignment → zero page-cache overhead
- Setup: `zfs set recordsize=1M pool/dataset`

---

## 4. GPU Dispatch Threshold

`VULKAN_MIN_ELEMS = 4_194_304` (4M elements, ~16MB F32, ~8MB F16)

In `src/tensor/mod.rs`. This constant controls the hybrid tile-pulling dispatcher:
- **Below**: only CPU SIMD runs. GPU thread is not spawned.
- **At/above**: GPU dispatcher thread competes for tiles alongside CPU.

To tune for faster PCIe (e.g., high-end discrete GPU), lower this constant.

---

## 5. Maximizing Throughput

1. **Small tensors (< 2M elements)**: Always use `device="cpu"`. Vulkan PCIe staging overhead is not amortized.

2. **F16/BF16 MatMul**: Use `device="vulkan"`. Even though the GPU computes in F32, OxTorch's upcast path is ~500× faster than PyTorch's CPU scalar path.

3. **Large activations (> 4M elements)**: Use `device="hybrid"`. MSTS tile-pulling lets GPU and CPU race for tiles in parallel.

4. **Out-of-core weights**: Use `Tensor.from_ssd()`. Never load multi-GB tensors with numpy if they don't fit comfortably in RAM below `l2_ram_max_bytes`.

5. **Reduce background processes** during benchmark runs to minimize StdDev noise.

---

## 6. Thermal Behavior

- **Cold start**: L3 cache empty; first runs slower.
- **Warm** (after a few minutes): tile sizes cache-resident, times stabilize.
- **Heat soak (2+ hours)**: CPU/GPU clock down 10–15%. OxTorch/PT ratio stays stable; both affected equally.

StdDev thresholds:
- < 5ms: excellent
- 5–50ms: normal (OS activity)
- > 50ms: thermal throttling or memory pressure

---

## 7. Known Limitations

- **Small GPU dispatches**: Bonaire ~80ms PCIe cost makes Vulkan a net negative below 4M elements. Hardware characteristic, not a bug.
- **No native FP16 SPIR-V compute**: GCN 1.1 shader units compute in F32. F16/BF16 is always converted on CPU side.
- **F16 MatMul Hybrid**: tile-pulling hybrid currently covers activations. MatMul hybrid uses full-tensor Vulkan dispatch (fast for matmul sizes).
- **INT8 GELU/Softmax — no PyTorch reference**: PyTorch has no native CPU INT8 GELU/Softmax kernel. OxTorch has native implementations. Parity is verified against a float32 reference.
- **autograd / training**: Not yet implemented. Inference-only (Sprint 7 long-term).
