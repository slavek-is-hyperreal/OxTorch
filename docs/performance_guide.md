# Performance Guide (v3.8.1-rc — The MSTS v2 Leap)

This guide explains how to interpret benchmark results and maximize throughput
on the reference hardware: i5-3450 (Ivy Bridge), AMD Radeon R7 200 (Bonaire GCN 1.1), 24GB DDR3, ZFS SSD pool.

---

## 1. Running Benchmarks

The primary benchmark system is the **Atomized Suite**:

```bash
source venv/bin/activate
# Standard run (all benchmarks)
python3 tests/run_all_benchmarks.py
```

Output format (one line per test):
```
Linear_bf16_cpu | PT: 6.8389s | VNN: 0.2599s | 0.0380x | ✅ PASS
```

---

## 2. Benchmark Results (v3.8.1-rc, i5-3450)

### Key Gains — BF16 is the "Sweet Spot"

| Test | PyTorch | OxTorch | Ratio | Why? |
|:---|---:|---:|:---:|:---|
| **Linear BF16 (cpu)** | 6.83s | 0.26s | **0.038x** 🚀 | **26x faster** vs PT scalar. |
| **LayerNorm BF16 (cpu)** | 2.0ms | 0.5ms | **0.25x** 🚀 | **4x faster** SIMD kernel. |
| **GELU BF16 (hybrid)** | 21.2ms | 10.9ms | **0.51x** ✅ | **2x faster** via MSTS racing. |
| **ReLU F16 (cpu)** | 132.6s | 0.17s | **~0.0013x** 🚀 | F16C intrinsics vs PT scalar. |

> **Why BF16 is so fast on IVY Bridge**: The i5-3450 has **AVX+F16C** but **not AVX-512**. 
> PyTorch's native BF16 requires **AVX-512**. Without it, it falls back to scalar emulation. 
> OxTorch implements high-performance **SSE2/AVX1 SWAR** and **F16C conversion**, making it orders of magnitude faster on legacy Intel chips.

---

## 3. Hardware Characteristics

### CPU (i5-3450, Ivy Bridge)
- 4 cores, no hyperthreading, 6MB L3 cache
- **Has**: AVX, F16C, SSE4.1 — **No**: AVX2, FMA, AVX-512
- **F16/BF16 dispatch**: Uses F16C intrinsics (F16) or SSE2/AVX1 SWAR (BF16).
- **Reductions**: Uses `f64` accumulation threads to maintain 100% parity with PyTorch.

### GPU (AMD Radeon R7 200 Series, Bonaire GCN 1.1)
- ~1GB GDDR5 VRAM
- PCIe 3.0 — round-trip staging cost: ~80ms on Bonaire
- **GPU break-even**: **≥4M elements** (~16MB F32, ~8MB F16)
- Below threshold: The CPU is always faster due to PCIe overhead.

### SSD (ZFS/NVMe Environment)
- `io_uring` + `O_DIRECT` + **4096-byte recordsize alignment**.
- **MSTS Strategy**:
    - `DIRECT_MAX`: 50% of L3 cache size (approx 3MB for i5-3450).
    - `TILE_LARGE`: Fixed 8MB for high-speed sequential I/O.

---

## 4. Maximizing Throughput

1. **Small tensors (< 1M elements)**: Always use `device="cpu"`. 
2. **BF16 LLM Inference**: Always use OxTorch. You will see a **10x-20x** speedup over PyTorch on non-AVX512 CPUs.
3. **Out-of-RAM Datasets**: Use `Tensor.from_ssd()`. MSTS v2 will stream the weights directly into the CPU registers, allowing 100GB+ models to run on 16GB of RAM.

---

## 5. Known Limitations (v3.8.1-rc)

- **PCIe overhead on GCN 1.1**: The 80ms latency of the R7 200 makes small GPU dispatches a net negative.
- **Python-side Indentation**: (Under Fix) Ensure `oxtorch/tensor.py` is correctly indented to allow the `__getattr__` fallback to fire properly.
- **Autograd / Training**: Not yet implemented. Inference-only.
