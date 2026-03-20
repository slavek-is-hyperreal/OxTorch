# Roadmap - OxTorch v3.7.0

## [3.7.0] The BitNet Leapfrog & Rebranding (LATEST)
- [x] **OxTorch Rebranding**: Global project transition and documentation overhaul.
- [x] **BitNet 1.58b**: Native `Ternary` types and dequantization-free kernels.
- [x] **F16 CPU Parallelization**: Performance parity on legacy non-AVX512 hardware.
- [x] **100% Bit-Perfect BitNet**: Parity verified between CPU and Vulkan backends.
- [x] **100% PyTorch Fallback Dispatcher**: `import oxtorch as torch` drop-in replacement via `oxtorch/` package.
- [x] **Phase 6 — Atomized Benchmark Suite**: 105 self-contained benchmark files (BF16/F16/F32/INT8/Monster), each saving results to `tests/results/*.json`. OxTorch faster in **34/53** completed tests. MatMul Vulkan: **4–25x** faster than PyTorch across all dtypes.


## [3.6.0] Strategy: Hardware Acceleration & High-Precision Reductions

---

## 🚀 Active Development: Sprint 1.6 "Divide & Conquer"
*Goal: Re-board the engine for 100% parity across all devices through modular isolation and PyTorch fallbacks.*

### 1. Architectural Re-boarding
- [x] **Modular Directory Structure**: Transitioned to `src/{cpu,vulkan}/ops/`.
- [x] **PyTorch Fallback Mechanism**: Implemented `fallback.rs` for 100% API coverage.
- [x] **Strict Hybrid Validation**: Deferred to Sprint 4 (performance hardening phase).

### 2. Implementation & Migration
- [x] **Migrate MatMul and Softmax** to modular directory structure.
- [x] **Parallel Dispatchers** for F16 scalar CPU fallbacks.
- [x] **Vulkan BitLinear**: High-efficiency shader for ternary weights.

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
*Target: run LLaMA/Mistral/Phi mini inference (quantized, SSD-resident).*
- [ ] **Matrix**: `bmm` (Batch MatMul), `outer` (RoPE).
- [ ] **Fused Linear**: `F.linear(x, W, b)` - mm + bias + relu in one Vulkan dispatch.
- [ ] **Normalization**: `layer_norm`, `rms_norm` (LLaMA style).
- [ ] **Sequence**: `cat`, `stack`, `split`, `chunk`.
- [ ] **Indexing**: `index_select` (gather/scatter), `__getitem__` (slicing).
- [ ] **Embeddings**: `embedding(input, weight)` lookup table.
- [ ] **Attention**: `scaled_dot_product_attention` (fused Vulkan mega-kernel).
- [ ] **Decoding**: `argmax`, `topk`.
- [/] **MSTS PyTorch Fallback** — generalize tile-pulling to arbitrary `Callable[[np.ndarray], np.ndarray]`. Enables: (1) SSD streaming for any op without a native Vulkan shader (e.g. `layer_norm`, `erf`, `embedding`), (2) memory-efficient processing of tensors larger than RAM by materializing only 256K-element tiles at a time.

### [Sprint 2.5] CPU Architecture Retrofits (SIMD Expansion)
*Target: Systemic hardware optimization for non-AVX desktop environments.*
- [ ] **ARM64 NEON**: Manual wektoryzacja `vaddq_f32`, `vmulq_f32` dla `layer_norm` i `rms_norm`.
- [ ] **AVX-512**: Obsługa 512-bitowych rejestrów `zmm` dla najnowszych procesorów Intel/AMD.
- [ ] **SSE2 Fallback**: Dedykowane pętle 128-bitowe dla starszych jednostek x86_64.
- [ ] **F16C Hardware Acceleration**: Wykorzystanie `_mm256_cvtps_ph` do szybkiej konwersji typów na CPU.

### [Sprint 2.1] BitNet (1.58b) - The LLM Leapfrog
*Target: native support for 1.58-bit ternary models (Bielik, BitNet-7B).*
- [x] **BitLinear Layer**: Custom CPU (parallel) and Vulkan kernels.
- [x] **1.58b Quantization**: Ternary weights `{-1, 0, 1}` integrated into `Tensor` constructors.
- [ ] **Bielik-to-BitNet**: Conversion scripts for Oxido's LLM ecosystem (On Deck).
- [x] **Zero-Multiplication Inference**: Efficient accumulation path for ternary weights.

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

### [Sprint 4] Performance: Fused Kernels & AVX1
*Target: match or beat PyTorch on modern hardware, dominate on legacy.*
- [ ] **Fused MatMul+Bias+ReLU** — Vulkan mega-kernel.
- [ ] **Fused LayerNorm** — single Vulkan dispatch.
- [ ] **Fused Attention** — FlashAttention-style tiled QKV.
- [x] **AVX1 `vmaxps` kernel for ReLU** (DONE).
- [x] **AVX1 vectorized `exp`** (DONE).
- [ ] **Buffer pool Drop integration**.
- [ ] **Descriptor set caching**.
- [ ] **MSTS Pre-flight Validation**: `MSTS` pre-flight checks for CPU+Vulkan dual-device support before hybrid dispatch.
- [ ] **Tagged-Token Dataflow**: Evolve MSTS from AtomicU32 to full TTDF tag-matching (MERA-400 P/Q flags).
- [ ] **Cooperative Matrix GLSL Shader**: `KHR_cooperative_matrix` for Tensor Cores.

### [Sprint 5] Dtype & Device Ergonomics
*Target: full API compatibility — any PyTorch code compiles with 1 import change.*
- [ ] `Tensor.to(dtype)` — runtime casting.
- [ ] `Tensor.to(device)` — CPU/Vulkan/Hybrid migration.
- [ ] `Tensor.clone()`, `Tensor.contiguous()`.
- [ ] Automatic Broadcasting alignment.
- [ ] `torch.save` / `torch.load` pickle compatibility.

- [ ] **GGUF Support**: `Q4_K`, `Q8_0`, `Q6_K` block-quantized formats.
- [ ] On-the-fly dequantization inside SPIR-V registers.

### [Sprint 7] Training (Long Term)
*Target: OxTorch as a training engine.*
- [ ] Autograd: `requires_grad`, `.backward()`, gradient tape.
- [ ] Optimizer primitives: SGD, Adam.
- [ ] Loss functions: `cross_entropy`, `mse_loss`.

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
