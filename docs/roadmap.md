# Roadmap - OxTorch v3.7.0

## [3.7.0] The BitNet Leapfrog & Rebranding (LATEST)
- [x] **OxTorch Rebranding**: Global project transition and documentation overhaul.
- [x] **BitNet 1.58b**: Native `Ternary` types and dequantization-free kernels.
- [x] **F16 CPU Parallelization**: Performance parity on legacy non-AVX512 hardware.
- [x] **100% Bit-Perfect BitNet**: Parity verified between CPU and Vulkan backends.
- [x] **100% PyTorch Fallback Dispatcher**: `import oxtorch as torch` drop-in replacement via `oxtorch/` package.
- [x] Phase 6 — Atomized Benchmark Suite: 167 self-contained benchmark files (BF16/F16/F32/INT8/Monster), each saving results to tests/results/*.json. OxTorch faster in **88/167** tests. MatMul CPU (BF16): **400–700x** faster than PyTorch fallback.


## [3.6.0] Strategy: Hardware Acceleration & High-Precision Reductions

---

## 🚀 Active Development: Sprint 2 "Transformers Ops — LLM Ready" [/]
*Goal: Run LLaMA/Mistral/Phi mini inference (quantized, SSD-resident). 167/167 benchmarks passing.*

### Status: Sprint 1.6 ✅ COMPLETE
- [x] **Modular Directory Structure**: `src/{cpu,vulkan}/ops/`.
- [x] **PyTorch Fallback Mechanism**: `fallback.rs` — 100% API coverage.
- [x] **Vulkan BitLinear**: High-efficiency shader for ternary weights.
- [x] **163/163 parity tests passing**, 67 OxTorch-faster, MatMul Vulkan up to 700× faster than PyTorch.

### In Progress
- [x] **MSTS PyTorch Fallback** ✅ — Generalized tile-pulling in `oxtorch/tensor.py`.
- [x] `bmm` (Batch MatMul) [NATIVE] ✅.
- [ ] `outer` (RoPE) — (Pending native implementation).
- [ ] `scaled_dot_product_attention` — [FALLBACK], fused Vulkan mega-kernel pending.
- [ ] `index_select`, `__getitem__` (slicing), `embedding` lookup — [FALLBACK].
- [ ] `argmax`, `topk` (decoding ops) — [FALLBACK].

---

## 🗺️ Long-Term Roadmap (Divide & Conquer Aligned)

### [Sprint 1] Foundation Ops (MLP Ready)
*Target: Refactor existing CPU/Vulkan ops into the Sprint 1.6 modular structure.*
- [x] `mul` / `sub` / `div` elementwise arithmetic.
- [x] `scalar_mul` / `scalar_add` broadcast — `tensor * 2.0`, `tensor + 1.0` (Native SIMD).
- [x] `reshape`, `view`, `squeeze`, `unsqueeze`, `flatten`.
- [x] `gelu`, `leaky_relu`, `elu`, `tanh`, `clamp`.
- [x] `sum`, `mean`, `max`, `min` (reduce ops).
- [x] `softmax`, `log_softmax`.
- [x] `zeros`, `ones`, `full`, `rand`, `randn` (creators).

### [Sprint 2] Transformers Ops (LLM Ready) [/]
*Target: native implementation of LLaMA/Mistral/Phi inference ops.*
- [x] **Matrix**: `bmm` ✅. remaining: `outer` (RoPE).
- [ ] **Fused Linear**: `F.linear(x, W, b)` - mm + bias + relu in one Vulkan dispatch.
- [x] **Normalization**: `rms_norm` ✅, `layer_norm` ✅.
- [x] **Sequence**: `cat` ✅, `stack` ✅, `split` ✅, `chunk` ✅.
- [ ] **Indexing**: `index_select`, `__getitem__` (Pending native implementation).
- [ ] **Embeddings**: `embedding` lookup (Pending native implementation).
- [ ] **Attention**: `scaled_dot_product_attention` (Pending native implementation).
- [ ] **Decoding**: `argmax`, `topk` (Pending native implementation).
- [x] **Arithmetic**: `div`, `sum`, `mean`, `max` ✅. remaining: `mod`, `pow`.
- [x] **MSTS PyTorch Fallback**: `msts_pytorch_apply` ✅ (Infrastructure only).

**Performance regressions (fixed 2026-03-21):**
- [x] **`sub_i8_swar` is scalar stub** — implemented SSE2 path in `sub_i8.rs`.
- [x] **Per-op `Vec<u8>` allocation** — Fixed with `TensorPool` slab allocator ✅.
- [x] **ScalarAdd/ScalarMul f16 SIMD gap** — vectorized `scalar.rs` with AVX2/F16C. Up to 12x faster.
- [x] **AlignmentMismatch Crisis** — Eliminated all `bytemuck` panics via `f64` pool + manual `unsafe` casting ✅.
- [x] **Benchmark artifact: `split` parity overhead** — optimized `base.py` and benchmark params.

**Monster Test Framework** *(validates the core MSTS promise)*:
- [x] **`MonsterBenchmark` base class** — [DONE].
- [x] **Per-op Monster variants** — `Monster_ReLU_F32_SSD` [DONE].
- [ ] **Parity strategy for Monster** — partial verification.
- [x] **Metric: MB/s throughput** — [DONE].
- [x] **Auto-skip on non-MSTS builds** — [DONE].

### [Sprint 2.5] CPU Architecture Retrofits (SIMD Expansion)
*Target: Systemic hardware optimization for non-AVX desktop environments.*
- [ ] **ARM64 NEON**: Manual wektoryzacja `vaddq_f32`, `vmulq_f32` dla `layer_norm` i `rms_norm`.
- [ ] **AVX-512**: Obsługa 512-bitowych rejestrów `zmm` dla najnowszych procesorów Intel/AMD.
- [ ] **SSE2 Fallback**: Dedykowane pętle 128-bitowe dla starszych jednostek x86_64.
- [ ] **F16C Hardware Acceleration**: Wykorzystanie `_mm256_cvtps_ph` do szybkiej konwersji typów na CPU.

### [Sprint 2.1] BitNet (1.58b) — The LLM Leapfrog + LUT-GEMM Supremacy
*Target: native support for 1.58-bit ternary models. Beat Microsoft BitNet repo on x86 CPU and GPU.*

**Borrowed from BitNet repo (and improved):**
- [ ] **`DataType::Ternary2bpp`** — packed 2-bit-per-weight storage (vs. current i8-per-weight). Reduces weight memory footprint by 4x.
- [ ] **LUT-GEMM CPU kernel** (`bitlinear_lut_f32`) — replaces multiply-accumulate with LUT index lookups, matching BitNet TL2 approach but in native Rust with Rayon parallelism.
- [ ] **AVX2/AVX-512 LUT-GEMM** — vectorized LUT lookup using `_mm256_i32gather_ps` / `_mm512_i32gather_ps`. Faster than BitNet's C++ for matching micro-arch (via hardware-native wheel compilation).

**Our secret sauce (BitNet repo has none of this):**
- [ ] **MSTS-native BitNet streaming** — stream 70B ternary weights tile-by-tile from SSD through MSTS ring buffer. Enables BitNet inference on machines with 8GB RAM.
- [ ] **Vulkan LUT-GEMM shader** — LUT-GEMM on GPU via Vulkan compute. BitNet repo is CPU-only.
- [ ] **Hybrid dispatch** — CrookScheduler splits weight tiles between CPU LUT-GEMM and Vulkan LUT-GEMM simultaneously.

**Correct normalization (required for non-English BitNet):**
- [ ] **SubLN kernel** — Sub-LayerNorm (pre + post normalization per block) instead of plain RMSNorm. Without SubLN, ternary weights drift toward zero for morphologically rich languages (Polish, Czech) due to asymmetric rounding accumulation. Required by Sprint 10.

**Already done:**
- [x] **BitLinear Layer**: Custom CPU (parallel) and Vulkan kernels.
- [x] **1.58b Quantization**: Ternary weights `{-1, 0, 1}` integrated into `Tensor` constructors.
- [x] **Zero-Multiplication Inference**: Efficient accumulation path for ternary weights.
- [ ] **Bielik-to-BitNet**: Conversion scripts for Oxido's LLM ecosystem → see Sprint 10.

### [Sprint 3] CNN & Vision Models
*Target: run ResNet / EfficientNet / ViT inference.*
- [ ] `conv2d` (Winograd 3x3 optimized for Ivy Bridge).
- [ ] `conv1d` for sequence models.
- [ ] `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`.
- [ ] `upsample` / `interpolate`.

### [Sprint 3.5] Ultra-Legacy & 32-bit Retrofits
*Target: Execute on natively 32-bit constrained hardware (RPi1, Netbooks).*
- [ ] `cfg(target_pointer_width = "32")` cleanups.
- [ ] **Intel 80387 FPU Path**: Ternary-accumulation backends for non-SSE 32-bit processors.
- [ ] `Xoshiro128++` fallback for 32-bit PRNG.
- [ ] Strict `#UD` protection (Illegal Instruction) for non-SIMD processors.
- [ ] Sub-256MB VRAM/RAM pooling constraints.

### [Sprint 4] Performance: Fused Kernels, AVX1, and MERA-400 Optimizations
*Target: match or beat PyTorch on modern hardware, dominate on legacy. Prepare MSTS for distributed dispatch.*

**Fused Kernels (eliminate PCIe round-trips):**
- [ ] **Fused MatMul+Bias+ReLU** — Vulkan mega-kernel (single shader dispatch, not three).
- [ ] **Fused LayerNorm** — single Vulkan dispatch.
- [ ] **Fused Attention** — FlashAttention-style tiled QKV.
- [x] **AVX1 `vmaxps` kernel for ReLU** (DONE).
- [x] **AVX1 vectorized `exp`** (DONE).

**Buffer / Pipeline:**
- [x] **`TensorPool` slab allocator** — pre-allocate output buffers by size class and recycle them per-op. Eliminates the `Vec<u8>` allocation+deallocation overhead. (DONE 2026-03-21).
- [x] **Buffer pool Drop integration** (via `Storage::drop` manual cleanup).
- [ ] **Descriptor set caching**.
- [ ] **Static Computation Graph mode** — for fixed architectures (e.g. Copilot inference), pre-compile `VkCommandBuffer` once and replay per token. Eliminates dispatch overhead.

**MERA-400-Inspired MSTS upgrades** *(from `docs/experminental_plans/MERA-400_...md`)*:
- [x] **Dual-path MSTS dispatch** *(PULLED FORWARD FROM SPRINT 4, verified 2026-03-21)* — 3 compile-time paths based on tensor size. Eliminates thread spawn overhead for small/medium tensors:
  - **Path A (Direct):** ✅ tensor < `MSTS_DIRECT_MAX` → mmap read_exact + single-thread AVX, zero atomics.
  - **Path B (Single-thread MSTS):** ✅ < 32 MB → 1 IO worker, tile = `MSTS_TILE_SMALL` (≈ 75% L2), ring = 2. Stays in L2 cache during compute.
  - **Path C (Full CrookScheduler):** ✅ ≥ 32 MB → 2 workers + Rayon parallel, tile = `MSTS_TILE_BYTES` (4 MB for SATA burst).
  - Thresholds compiled in via `build.rs` reading L2/L3 sysfs (see `docs/binary_distribution.md` — Sprint 6 already has this infrastructure).
  - See `docs/implementation_guides.md` → "Sprint 4 — MSTS Dual-Path Dispatch" for full code.
- [ ] **Back-pressure instead of spin-loop** *(EN signal analog)* — tiles in `BUSY` state yield CPU time to the Vulkan thread instead of spinning.
- [ ] **Graceful GPU worker disconnect** *(OFF+ZF signal analog)* — on Vulkan context loss, worker waits for current tile to finish before detaching. No corrupt bus state.
- [x] **MSTS Pre-flight Validation**: CPU+Vulkan dual-device sanity check before hybrid dispatch (DONE).
- [ ] **Tagged-Token Dataflow (TTDF)**: Evolve MSTS from `AtomicU32` to full tag-matching (MERA-400 P/Q flags). *This is the prerequisite for Sprint 8.*
- [ ] **Cooperative Matrix GLSL Shader**: `KHR_cooperative_matrix` for Tensor Cores.

### [Sprint 5] Multi-Stream MSTS (Out-of-Core Operations)
**Goal:** Expand MERA-400 Tiling System (MSTS) beyond unary operations to support multi-tensor streaming. This enables true "beyond-RAM" processing for binary operations and reductions without OOM.

**Architectural Upgrades:**
- [ ] **DualStreamScheduler:** Extend `CrookScheduler` to manage two independent synchronized `io_uring` read queues.
- [ ] **Streaming Accumulator:** RAM-resident reduction buffers fed by SSD tiles.

**Key Deliverables:**
- [ ] **Elementwise Binary SSD:** Implement out-of-core `add`, `sub`, `mul`, `div` (Phase 1).
- [ ] **Streaming Reductions:** Implement `sum`, `mean`, `max`.
- [ ] **Broadcasting MSTS:** Support binary ops where Tensor B fits in RAM but Tensor A is SSD-resident.
- [ ] **Monster Binary Benchmarks:** Add monster benchmarks for `add` and `mul` comparing PyTorch swap vs VNN MSTS.

### [Sprint 6] Out-of-Core Linear Algebra & Strided I/O
**Goal:** Implement true out-of-core matrix multiplication and complex normalizations for trillion-parameter scale experimentation on consumer SSDs.

**Key Deliverables:**
- [ ] **Strided I/O Writer (`cat` dim>0):** Implement vectored `io_uring` (readv/writev) to scatter/gather blocks, enabling `cat`/`stack` on arbitrary dimensions without RAM materialization.
- [ ] **1-Pass LayerNorm:** Compute mean/variance on the fly per-row during streaming to avoid a 2-pass SSD read/write cycle.
- [ ] **Out-of-Core Linear ($C = A \times W$):** Block-matrix multiplication. 
  - Priority 1: $W$ fits in RAM (typical LLM inference).
  - Priority 2: Cannon's algorithm (neither fits in RAM).

### [Sprint 7] Dtype & Device Ergonomics
*Target: full API compatibility — any PyTorch code compiles with 1 import change.*
- [ ] `Tensor.to(dtype)` — runtime casting.
- [ ] `Tensor.to(device)` — CPU/Vulkan/Hybrid migration.
- [ ] `Tensor.clone()`, `Tensor.contiguous()`.
- [ ] Automatic Broadcasting alignment.
- [ ] `torch.save` / `torch.load` pickle compatibility.

- [ ] **GGUF Support**: `Q4_K`, `Q8_0`, `Q6_K` block-quantized formats.
- [ ] On-the-fly dequantization inside SPIR-V registers.

### [Sprint 8] Hardware-Native Binary Distribution
*Target: `pip install oxtorch` delivers a wheel compiled for user's exact CPU+GPU.*
- [ ] **`oxtorch-detect` CLI** (Rust) — collects CPU micro-arch, L2/L3 cache sizes, SIMD flags, GPU PCI ID, VRAM.
- [ ] **Build server** — receives hardware fingerprint, compiles with `-C target-cpu=<micro-arch>` and `MSTS_TILE_BYTES=<0.75*L2>`, caches wheel on S3/R2.
- [ ] **`build.rs` parametrization** — read `MSTS_TILE_BYTES`, `MSTS_RING_DEPTH`, `VNN_SUBGROUP_SIZE` from env at compile time instead of hardcoded constants.
- [ ] **Install script** — `curl -sSf https://get.oxtorch.io | bash` detects hardware, fetches hardware-specific wheel.
- See [`docs/binary_distribution.md`](file:///my_data/gaussian_room/docs/binary_distribution.md) for full design.

### [Sprint 9] Training (Long Term)
*Target: OxTorch as a training engine.*
- [ ] Autograd: `requires_grad`, `.backward()`, gradient tape.
- [ ] Optimizer primitives: SGD, Adam.
- [ ] Loss functions: `cross_entropy`, `mse_loss`.

### [Sprint 10] Distributed Cluster — "Horizontal Scalability for the Rest of Us"
*Target: N machines with mixed old GPU/CPU collectively run one model via TTDF tile dispatch over the network.*

This is the "500 laptops" vision: each machine runs `oxtorch-node`, claims tiles tagged with the operations it can handle, computes, returns results. No barriers, no static splits, no identical hardware requirement.

- [ ] **`oxtorch-node` daemon** — advertises capabilities (CPU micro-arch, GPU, VRAM) and consumes tiles from the network queue.
- [ ] **Network tile transport** — lightweight binary protocol over UDP (no gRPC/HTTP overhead). Tile header = 32-bit TTDF tag + 16-bit checksum.
- [ ] **TTDF over network** — same P/Q flag semantics as local MSTS, but tile state propagates across machines. Self-healing: unclaimed tiles re-queue after timeout.
- [ ] **Hot-plug nodes** — machine joins cluster mid-inference, picks up the next available tile.
- [ ] **BitNet 1.58b as transport format** — ternary weights packed at 2bpp (~16 weights per 32-bit word) minimize bandwidth. 1Gbps Ethernet becomes equivalent to ~10Gbps for weight transfers.
- [ ] **Virtual network test mode** — simulate N nodes in-process (each as a thread with throttled queue) to validate TTDF before touching real cables.
- Prerequisite: Sprint 4 TTDF + Sprint 2.1 `Ternary2bpp`.

### [Sprint 11] Local BitNet Copilot (LSP Server)
*Target: open-source, local, IDE-integrated code assistant running 100% on the user's hardware via BitNet 1.58b + MSTS.*

**Available 1.58b models (March 2026) — no conversion needed:**
| Model | Size | Notes |
|---|---|---|
| `microsoft/bitnet-b1.58-2B-4T` | 2B | Code+math, GGUF-ready. ~750MB VRAM. |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | 8B | Llama3 fine-tuned to 1.58b. |
| `tiiuae/Falcon3-10B-Base-1.58bit` | 10B | Falcon3 ternary. |
| `kgrabko/JiRackTernary_*` | 1B–405B | Community, incl. 70B+. |

**Competitive context:** Tether's **QVAC Fabric** (March 2026) adds Vulkan GPU kernels + dynamic tiling to `llama.cpp`. Benchmarks: 2.1–11.3× over CPU on edge GPUs, bit-exact with CPU, LoRA fine-tuning up to 13B on consumer hardware. This is our closest competitor on the Vulkan+tiling approach.

**OxTorch differentiators vs. QVAC Fabric:**
- SSD streaming via MSTS — 70B+ models on 8GB RAM with no external swap
- TTDF-ready tile scheduling (Sprint 4) — prerequisite for Sprint 8 cluster inference
- Hardware-native wheel (Sprint 8) — tiles/ring depth tuned to exact L2/L3 cache of user hardware
- `import oxtorch as torch` — zero code change for existing PyTorch code

- [ ] **LSP server** (`oxtorch-lsp`) — Rust binary exposing Language Server Protocol. IDE (VS Code, Neovim) talks to it via stdio.
- [ ] **Streaming token generation** — MSTS streams weight tiles from SSD, GPU computes, tokens stream back to IDE character by character.
- [ ] **Repo-as-context via MSTS** — index entire codebase on SSD; MSTS streams relevant chunks into the 8K context window without loading into RAM.
- [ ] **Static dispatch graph** — fixed architecture: compile `VkCommandBuffer` once at startup, replay per token. Eliminates per-token dispatch overhead.
- [ ] **GGUF loader for BitNet** — load any of the above models from GGUF natively.
- [ ] **First milestone: `bitnet-b1.58-2B-4T` running as Copilot** — easiest entry point, ~750MB VRAM, already GGUF.
- [ ] **Second milestone: `Falcon3-10B-Base-1.58bit`** — 10B ternary, better quality, tests MSTS SSD streaming.
- Prerequisite: Sprint 2.1 LUT-GEMM + MSTS-native BitNet + Sprint 4 Static graph + Sprint 7 GGUF.


### [Sprint 12] Bielik → BitNet 1.58b Conversion
*Target: Polish-language LLM (Bielik 7B/11B) converted to 1.58b ternary weights via Knowledge Distillation + QAT, runnable on OxTorch.*
*(Research blueprint: `docs/experminental_plans/Kwantyzacja BitNet Bielik_ Fizyka Statystyczna.md`)*

**Key constraints vs. English models:**
- Polish is highly inflected → naive quantization destroys rare morphological tokens (declension endings). Standard `absmean` threshold collapses them to zero.
- Distillation temperature must be **T∈[3.0, 4.5]** (vs. T∈[1.0, 2.0] for English) to preserve the dense probability landscape over conjugated/declined forms.

**Pipeline:**
- [ ] **Importance Matrix** — compute gradient sensitivity (Hessian trace) of Bielik over a Polish corpus (SpeakLeash) to identify which weights encode rare morphology. These weights get protected during quantization.
- [ ] **Architecture surgery** — replace `torch.nn.Linear` with `BitLinear`, replace `RMSNorm` with `SubLN`. Keep Embeddings + LM Head in BF16 (32k–128k vocabulary cannot be ternary).
- [ ] **Composite loss function** — `L_total = L_CE + L_KLD(T=3.5) + λ·L_EP` where `L_EP` is an entropy penalty that prevents weight collapse to zero, with `λ=5×10⁻⁴`.
- [ ] **QAT + Simulated Annealing** — STE gradient estimator + stochastic thermal temperature `T_SA` (Ising model analog). Exponential cooling schedule over final 30% of training. Prevents local minima traps that STE alone cannot escape.
- [ ] **Export pipeline** — serialize to `Ternary2bpp` + GGUF for distribution, runnable on Sprint 11 LSP server as a Polish Copilot.
- Prerequisite: Sprint 2.1 SubLN + Sprint 2.1 Ternary2bpp + Sprint 7 GGUF.


---

## 🛠️ Technical Reference
For detailed, step-by-step implementation plans and SPIR-V shader code, see:
👉 [Implementation Guides](file:///my_data/gaussian_room/docs/implementation_guides.md)

---

## 📜 Version History
For a complete record of all technical changes and releases, see:
👉 [Changelog](file:///my_data/gaussian_room/docs/CHANGELOG.md)

---
*OxTorch: High-performance AI inference on legacy hardware. Constraints breed architecture.*
