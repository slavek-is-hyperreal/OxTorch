# Roadmap - VulkanNN Rusted

## [3.6.0] Strategy: Hardware Acceleration & High-Precision Reductions (STABLE)
- [x] **MSTS**: SSD-to-CPU-to-GPU streaming.
- [x] **Int8 SWAR**: Bit-parallel logic for Int8 legacy CPUs.
- [x] **Safe 64-bit Reductions**: `i64` exact summation for 4B+ elements (beats PyTorch).
- [x] **SIMD Softmax**: Fully vectorized 3-pass kernels (AVX512, AVX2, AVX, SSE).
- [x] **Dynamic Upcasting**: `Int8` ops output `F32` to prevent saturation at 127.

---

## 🚀 Active Development: Sprint 1.6 "Divide & Conquer"
*Goal: Re-board the engine for 100% parity across all devices through modular isolation and PyTorch fallbacks.*

### 1. Architectural Re-boarding
- [ ] **Modular Directory Structure**: Transition to `src/{cpu,vulkan}/ops/{op_name}/{target}/`.
    - `target/` on CPU: `no-avx`, `avx1`, `avx2`, `arm-neon`, `arm32`.
    - `target/` on Vulkan: `generic`, `subgroup`, `coop_matrix`.
- [ ] **PyTorch Fallback Mechanism**: Implement `vnn.fallback` (PyO3) to ensure 100% API coverage during development.
- [ ] **Strict Hybrid Validation**: `MSTS` strictly checks for dual-device support before allowing `Hybrid` mode.

### 2. Implementation & Migration
- [ ] Migrate **MatMul** and **Softmax** to the new modular structure.
- [ ] Implement parallel dispatchers for out-of-place activations (`relu_f32`, `gelu_f32`).
- [ ] Add **Vulkan Elementwise** (`mul`, `sub`, `div`) using the new single-shader pattern.

---

## 🗺️ Long-Term Roadmap (Divide & Conquer Aligned)

### [Sprint 1] Foundation Ops (MLP Ready)
*Target: Refactor existing CPU/Vulkan ops into the Sprint 1.6 modular structure.*
- [x] `mul` / `sub` / `div` elementwise arithmetic.
- [x] `scalar_mul` / `scalar_add` broadcast.
- [x] `reshape`, `view`, `squeeze`, `unsqueeze`, `flatten`.
- [x] `gelu`, `leaky_relu`, `elu`, `tanh`, `clamp`.
- [x] `sum`, `mean`, `max`, `min` (reduce ops).
- [x] `softmax`, `log_softmax`.
- [x] `zeros`, `ones`, `full`, `rand`, `randn` (creators).

### [Sprint 2] Transformers Ops (LLM Ready)
*Target: run LLaMA/Mistral/Phi mini inference (quantized, SSD-resident).*
- [ ] **Matrix**: `bmm` (Batch MatMul), `outer` (RoPE).
- [ ] **Fused Linear**: `F.linear(x, W, b)` - mm + bias + relu in one Vulkan dispatch.
- [ ] **Normalization**: `layer_norm`, `rms_norm` (LLaMA style).
- [ ] **Sequence**: `cat`, `stack`, `split`, `chunk`.
- [ ] **Indexing**: `index_select` (gather/scatter), `__getitem__` (slicing).
- [ ] **Embeddings**: `embedding(input, weight)` lookup table.
- [ ] **Attention**: `scaled_dot_product_attention` (fused Vulkan mega-kernel).
- [ ] **Decoding**: `argmax`, `topk`.

### [Sprint 3] CNN & Vision Models
*Target: run ResNet / EfficientNet / ViT inference.*
- [ ] `conv2d` (Winograd 3x3 optimized for Ivy Bridge).
- [ ] `conv1d` for sequence models.
- [ ] `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`.
- [ ] `upsample` / `interpolate`.

### [Sprint 3.5] Ultra-Legacy & 32-bit Retrofits
*Target: Execute on natively 32-bit constrained hardware (RPi1, Netbooks).*
- [ ] `cfg(target_pointer_width = "32")` cleanups.
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
- [ ] **Tagged-Token Dataflow**: Evolve MSTS from AtomicU32 to full TTDF tag-matching (MERA-400 P/Q flags).
- [ ] **Cooperative Matrix GLSL Shader**: `KHR_cooperative_matrix` for Tensor Cores.

### [Sprint 5] Dtype & Device Ergonomics
*Target: full API compatibility — any PyTorch code compiles with 1 import change.*
- [ ] `Tensor.to(dtype)` — runtime casting.
- [ ] `Tensor.to(device)` — CPU/Vulkan/Hybrid migration.
- [ ] `Tensor.clone()`, `Tensor.contiguous()`.
- [ ] Automatic Broadcasting alignment.
- [ ] `torch.save` / `torch.load` pickle compatibility.

### [Sprint 6] Quantization & GGUF
*Target: run quantized LLMs that don't fit in F16.*
- [ ] **INT8 quantization** — symmetric per-tensor.
- [ ] **GGUF Support**: `Q4_K`, `Q8_0`, `Q6_K` block-quantized formats.
- [ ] On-the-fly dequantization inside SPIR-V registers.

### [Sprint 7] Training (Long Term)
*Target: VulkanNN as a training engine.*
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
*VulkanNN: High-performance AI inference on legacy hardware. Constraints breed architecture.*
