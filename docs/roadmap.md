# Roadmap

---

## Completed

**v3.3.0 "Iron Age"** (dev_raw_vulkan branch)

- Raw `ash` Vulkan 1.2 backend replacing `wgpu`: explicit command pools, Timeline Semaphores,
  buffer recycling cache, separate compute and transfer queues.
- SPIR-V shader compilation at build time via `naga`.
- MSTS Tile-Pulling Hybrid Dispatch (Phase 4): `AtomicUsize` tile counter shared between GPU
  dispatcher and CPU SWAR threads. No locks. No static splits. 
  Inspired by the MERA-400 CROOK OS tagged-token dataflow architecture.
  See: [MERA-400 Wikipedia](https://pl.wikipedia.org/wiki/Mera_400) |
       [mera400.pl](https://mera400.pl/Strona_g%C5%82%C3%B3wna) |
       [mera400 YouTube channel](https://www.youtube.com/c/mera400)
- GPU dispatch threshold (4M elements): skips Vulkan on small tensors where PCIe overhead
  dominates.
- Cross-platform SIMD fallback chain (avx_swar.rs):
  F16C+AVX -> SSE2 SWAR -> AArch64 NEON -> scalar, for all 4 conversion directions.
- io_uring + O_DIRECT SSD streaming at 1MB ZFS recordsize boundaries.
- MERA Style Task Scheduler: StatefulTile lockless ring buffer in crook_scheduler.rs.
- Tri-precision engine: F32, F16, BF16 with PyTorch numerical parity across all modes.
- Statistical benchmark harness: multi-run Median/StdDev/Ratio tracking.
- Branch-specific module naming for A/B benchmarking across branches.
- Branch-aware dynamic import in unified_benchmark.py.

**v3.2.0 "Valkyrie"**

- Tri-precision engine (F32, F16, BF16)
- Statistical audit harness
- Session duration tracking for thermal analysis

**v2.9.0 - v2.8.0**

- CPU near-parity with PyTorch for RAM-resident F32 operations
- Unified benchmark harness
- Async triple-buffering pipeline

---

## In Progress

**Hybrid MatMul Tile-Pulling**

The tile-pulling Phase 4 dispatcher currently covers activation functions only.
Matrix multiplication still dispatches the full tensor to Vulkan in a single operation.
Phase 5 will extend the same AtomicUsize tile-pulling model to MatMul, allowing the CPU
sgemm path and the Vulkan shader to process row-tiles concurrently.

**Long-Term Stress Testing**

Validating SSD wear behavior and thermal clock-down over 5000+ iteration runs.
The benchmark history log is used to track cumulative thermal drift across sessions.

---

## Planned

**Gemma / LLM Inference Primitives**

- `Tensor::slice()` and `Tensor::view()` for slicing attention heads and KV caches
- `Tensor::concat()` for sequence concatenation
- These are required before any LLM forward pass can be expressed cleanly

**Operator Fusion in Vulkan Shaders**

Per the MERA-400 / Cray-1 vector chaining inspiration: instead of running a separate Vulkan
dispatch per operation (MatMul, then Bias, then ReLU), fuse multiple operations into a single
shader kernel so intermediate results stay in GPU register files without a PCIe round-trip.
This is especially important on Bonaire where the PCIe cost is the dominant bottleneck.

**GGUF Support**

Full support for GGUF headers and block-quantized memory formats (Q4_K, Q8_0, etc.),
enabling direct loading of weights from standard llama.cpp model files.

**INT8 / INT4 Quantization**

On-the-fly dequantization inside SPIR-V shader registers rather than on the CPU.
Goal: run Q4 models where even F16 weights would not fit in 1GB VRAM.

**Distributed VNN**

Multi-machine tensor sharding over gRPC for networked consumer hardware clusters.

**Asymmetric Speculative Decoding**

A two-engine pipeline where a small draft model resident in VRAM validates tokens from
a primary model streaming from SSD. Draft model runs ahead; mismatches cause backtrack.

---

*This library exists because AI inference should be possible on the hardware that most of the
world actually has access to. The MERA-400 ran a distributed operating system on components with
varying timing characteristics in 1976. Constraints breed architecture.*
