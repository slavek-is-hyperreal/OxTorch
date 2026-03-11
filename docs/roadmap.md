# Roadmap

---

## Completed

**v3.4.0 "Iron Age Complete"** (main)

- Merged experimental raw Vulkan 1.2 backend into `dev`.
- Removed `wgpu` dependency completely.
- Implemented dynamic VRAM footprint limits (512MB max) with a "Retry-on-OOM" allocator.
- Added explicit GC management for 15M element benchmarks in Python.
- **Performance bug fixes (6):** permanent Vulkan descriptor sets (no per-dispatch alloc/free),
  prune_buffer_cache O(n)→O(1), Rayon threshold 1M→8M, N CPU workers in hybrid scope,
  pre-allocated `a_band` buffers in `cpu_sgemm_streamed_f32`.
- **Free-list BufPool** (`buf_pool.rs`): zero-copy Vec<f32> reuse for hot tensor output paths.
- **Benchmark v3.3**: `relu_into` scenario, PyTorch backend labels (CPU·MKL·dtype),
  CPU Die temperature persistence to JSON, line-buffered output (flush=True).

**v3.3.0 "Iron Age"** (Experimental branch)

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
Phase 5 will extend the same AtomicUsize tile-pulling model to MatMul, allowing the CPU
sgemm path and the Vulkan shader to process row-tiles concurrently.

**Long-Term Stress Testing**

Validating SSD wear behavior and thermal clock-down over 5000+ iteration runs.
The benchmark history log is used to track cumulative thermal drift across sessions.

---

## PyTorch Parity Roadmap

The long-term goal is a **drop-in replacement for PyTorch**: change one import line and
existing PyTorch inference code runs — faster on legacy hardware, competitively on modern.

```python
# Before:
import torch
x = torch.tensor(data).cuda()

# After (VulkanNN drop-in):
import vulkannn as torch         # ← only change
x = torch.tensor(data).to("vulkan")
```

Each sprint below is a self-contained block of capability. Completing Sprint 1 means an
MLP forward pass works. Completing Sprint 2 means transformer inference works.

---

### Sprint 1 — "MLP Forward Pass" (Foundation ops)
*Target: run any feedforward network end-to-end*

#### Core Arithmetic
- [ ] `mul` — elementwise multiply (`*` operator), all dtypes, CPU+Vulkan+Hybrid
- [ ] `sub` — elementwise subtract (`-` operator)
- [ ] `div` — elementwise divide (`/` operator)
- [ ] `scalar_mul` / `scalar_add` — broadcast scalar over tensor (e.g. `x * 2.0`)

#### Shape Manipulation
- [ ] `reshape` / `view` — reinterpret shape without copy
- [ ] `squeeze(dim)` / `unsqueeze(dim)` — remove/add size-1 dimensions
- [ ] `flatten(start, end)` — flatten a range of dims into one

#### Activations
- [ ] `gelu` — Gaussian Error Linear Unit (dominant in GPT/BERT/LLaMA)
- [ ] `leaky_relu(negative_slope)` — for detection/GAN models
- [ ] `elu` — Exponential Linear Unit

#### Reductions
- [ ] `sum(dim)` — sum along axis (or whole tensor)
- [ ] `mean(dim)` — mean along axis
- [ ] `max(dim)` / `min(dim)` — max/min values

#### Functional Ops
- [ ] `softmax(dim)` — implemented as exp → sum → div (reuses above ops)
- [ ] `log_softmax(dim)` — for cross-entropy

#### Tensor Creation
- [ ] `Tensor.zeros(shape, dtype, device)` — without going via numpy
- [ ] `Tensor.ones(shape, dtype, device)`
- [ ] `Tensor.full(shape, fill_value, dtype, device)`

#### Implementation approach:
All Sprint 1 ops follow the same pattern as `__add__`:
1. CPU path: Rayon par_iter (or serial for small n via BufPool)
2. Vulkan path: new minimal SPIR-V shader per op (reuse pipeline pattern from activation shaders)
3. Hybrid path: MSTS tile-pulling dispatcher (same AtomicUsize counter model)
4. Vulkan shaders for `mul`/`div` are nearly identical to the existing `add` shader — minimal effort

---

### Sprint 2 — "Transformer Inference" (Attention & normalization)
*Target: run LLaMA/Mistral/Phi mini inference (quantized, SSD-resident)*

#### Matrix Ops
- [ ] `bmm` — batch matrix multiply `(B, M, K) @ (B, K, N) → (B, M, N)` (multi-head attention)
- [ ] `F.linear(x, W, b)` — fused `x @ W.T + b` in single Vulkan dispatch (Cray-1 chaining!)
- [ ] `outer(a, b)` — outer product (RoPE embeddings)

#### Normalization
- [ ] `layer_norm(x, weight, bias, eps)` — Transformer standard
- [ ] `rms_norm(x, weight, eps)` — LLaMA/Mistral standard (simpler than LayerNorm)
- [ ] `batch_norm` — CNN normalization

#### Sequence / Indexing
- [ ] `cat(tensors, dim)` — concatenate along axis (KV cache append)
- [ ] `stack(tensors, dim)` — stack new axis
- [ ] `split(tensor, size, dim)` — split into chunks (attention heads)
- [ ] `chunk(n, dim)` — split into N equal parts
- [ ] `index_select(dim, index)` — gather rows/cols by index tensor
- [ ] `__getitem__` — slicing: `x[0, :, 2:5]` Python subscript

#### Embeddings
- [ ] `embedding(input, weight)` — lookup table (every LLM needs this)

#### Attention
- [ ] `scaled_dot_product_attention(q, k, v, mask)` — composite op:
  implemented as `bmm(q, k.T) * scale → softmax → bmm(result, v)`
  Fused Vulkan mega-kernel per the Cray-1 vector chaining principle.

#### Decode
- [ ] `argmax(dim)` — greedy decoding
- [ ] `topk(k, dim)` — top-k sampling

#### Implementation approach:
- `bmm` extends the existing `__matmul__` with a batch loop + MSTS tile-pulling across batch dim
- `F.linear` = fused MatMul+Bias shader (one Vulkan dispatch, no PCIe roundtrip for intermediate)
- `scaled_dot_product_attention` fuses 4 ops into 1 shader — biggest single Vulkan perf gain possible
- Slicing/indexing on CPU path only first; GPU path if needed

---

### Sprint 3 — "CNN & Vision Models"
*Target: run ResNet / EfficientNet / ViT inference*

- [ ] `conv2d(input, weight, bias, stride, padding, dilation)` — Winograd algorithm for Ivy Bridge
- [ ] `conv1d` — for sequence models
- [ ] `max_pool2d` / `avg_pool2d`
- [ ] `adaptive_avg_pool2d`
- [ ] `upsample` / `interpolate` — bilinear, nearest
- [ ] `clamp(min, max)` — activation clipping (used in MobileNet etc.)
- [ ] `abs`, `sqrt`, `pow`, `exp`, `log` — full math elementwise library

#### Implementation approach:
- Conv2D: implement as Winograd 3x3 first (most common kernel size in modern CNNs)
  Per deep_research_on_optimization.md §3.2: tile to fit in 6MB L3 cache, Morton-order layout
- Vulkan shader for Winograd transform + elementwise multiply + inverse transform

---

### Sprint 4 — "Performance: Fused Kernels & AVX1"
*Target: match or beat PyTorch on modern hardware, dominate on legacy*

- [ ] **Fused MatMul+Bias+ReLU** — Vulkan mega-kernel (Cray-1 vector chaining: no PCIe for bias/act)
- [ ] **Fused LayerNorm** — single Vulkan dispatch (mean+variance+normalize+scale+shift)
- [ ] **Fused Attention** — FlashAttention-style: tiled QKV, compute in VRAM without full materialization
- [ ] **AVX1 `vmaxps` kernel for ReLU** — replace scalar loop in `avx_swar.rs`
  (`_mm256_max_ps(x, zero)` = 8 floats/cycle, purely AVX1, works on Ivy Bridge)
- [ ] **AVX1 vectorized `exp`** — for softmax (polynomial approximation via `_mm256_*`)
- [ ] **Buffer pool Drop integration** — Rust `Drop` for Tensor returns Vec to BufPool automatically
- [ ] **Descriptor set caching** — permanent sets for all op types (extend current matmul/add/act)
- [ ] **Tagged-Token Dataflow** — evolve MSTS from AtomicU32 to full TTDF tag-matching
  (per §2.3 of deep_research_on_optimization.md: MERA-400 P/Q flags as data-readiness tokens)

---

### Sprint 5 — "Dtype & Device Ergonomics"
*Target: full API compatibility — any PyTorch code compiles with 1 import change*

- [ ] `Tensor.to(dtype)` — cast between F32/F16/BF16/INT8 at runtime
- [ ] `Tensor.to(device)` — migrate between "cpu" / "vulkan" / "hybrid"
- [ ] `Tensor.clone()` — deep copy
- [ ] `Tensor.contiguous()` — ensure C-contiguous layout (no-op if already contiguous)
- [ ] Broadcasting — automatic shape alignment for mismatched tensor ops
- [ ] `Tensor.fill_(val)` — in-place fill
- [ ] `Tensor.zero_()` — in-place zero
- [ ] `torch.tensor(data)` — module-level constructor (not just class constructor)
- [ ] `torch.from_numpy(arr)` — zero-copy from numpy
- [ ] `torch.save` / `torch.load` — pickle-compatible serialization

---

### Sprint 6 — "Quantization & GGUF"
*Target: run quantized LLMs that don't fit even in F16*

- [ ] **INT8 quantization** — symmetric per-tensor, dequantize in Vulkan shader registers
- [ ] **INT4/Q4_K blocks** — GGUF block-quantized format, compatible with llama.cpp
- [ ] **Q8_0 / Q6_K / Q5_K** — remaining GGUF quantization schemes
- [ ] On-the-fly dequantization inside SPIR-V: weights stay quantized in VRAM/SSD until compute
- [ ] `torch.quantize_per_tensor` / `torch.dequantize` compatibility layer

---

### Sprint 7 — "Training (Long Term)"
*Target: VulkanNN as a training engine — last because inference is the market*

- [ ] Autograd: `requires_grad=True`, `.backward()`, gradient tape
- [ ] Optimizer primitives: SGD, Adam (elementwise + momentum buffers)
- [ ] Loss functions: `F.cross_entropy`, `F.mse_loss`, `F.binary_cross_entropy`
- [ ] `torch.no_grad()` context manager
- [ ] `DataLoader`-compatible dataset iteration

#### Note on training:
Training on constrained hardware (Ivy Bridge + 1GB VRAM) is genuinely hard — gradient
accumulation and mixed-precision training are the only practical paths. We approach this
following the same MSTS philosophy: tile the gradient accumulation across SSD-resident
parameter blocks, streaming gradients through the 1GB VRAM window.

---

## Other Planned Items (from original roadmap, unchanged)

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
