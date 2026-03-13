# Roadmap - VulkanNN Rusted

## [3.6.0] Hardware Acceleration & Tiling (STABLE)
- [x] **MSTS**: SSD-to-CPU-to-GPU streaming.
- [x] **Int8 SWAR**: Bit-parallel logic for Int8.
- [x] **Native PRNG**: Removal of Python-side random generation.
- [x] **Initial CPU Wins**: MatMul/Softmax beating PyTorch.

---

## [Sprint 2] Logic Fusion & Heterogeneous Compute
- [ ] **Multi-Device MSTS**: Support for simultaneous CPU + dGPU + iGPU execution.
- [ ] **Fused MatMul+Bias+ReLU**: Single-kernel SPIR-V for transformer linear blocks.
- [ ] **GELU Polynomial Kernels**: Sub-1.2x ratio on legacy CPUs.

---

## Completed

**v3.6.0 "Phase 3 — Hardware Acceleration & Int8 SWAR"** (dev)

- **Int8 SWAR**: 8-way parallel arithmetic for legacy CPUs using 64-bit GPRs.
- **Cache-Oblivious Tiling**: Recursive matrix multiplication strategy for CPU performance portability.
- **Subgroup Reductions**: `VK_KHR_shader_subgroup` extension enabled; reduction ops wired.
- **Cooperative Matrix** ⚠️: `VK_KHR_cooperative_matrix` extension *detection* enabled in `init_backend`. **The actual GLSL compute shader (`coop_matrix.comp`) is NOT yet written.** Real Tensor/Matrix Core utilization is a Sprint 4 item.
- **Backend Stability**: `UnsafeSendSync` implementation for asynchronous Vulkan execution.

**v3.5.0 "Sprint 1 — MLP Forward Pass"** (main)

- **Completed Sprint 1**: All foundation operations implemented with tri-precision (F32, F16, BF16) CPU/Vulkan/Hybrid paths.
- **Core Arithmetic**: `mul`, `sub`, `div`, broadcasted scalar arithmetic.
- **Shape Manipulation**: `reshape`, `view`, `squeeze`, `unsqueeze`, `flatten`.
- **Activations**: `gelu`, `leaky_relu`, `elu`, `tanh`, `clamp`.
- **Reductions & Functional**: `sum`, `mean`, `max`, `min`, `softmax`, `log_softmax`, including tree-reduction and shared memory kernel paths.
- **Tensor Creation**: `zeros`, `ones`, `full`, `rand`, `randn` directly bridging to NumPy.
- **Benchmark Update**: Added full Softmax, Elementwise, and Activation sweeps.

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

### Sprint 1.5 — "Extreme Engine Optimization" (Audit Debt)
*Target: Align Sprint 1 ops with `deep_research_on_optimization.md` axioms. All items below have a measurable success criterion: the named benchmark ratio must drop below 1.5x vs. PyTorch.*

---

#### ✅ DONE
- [x] **Int8 SWAR (Legacy CPU Master)**: 8-bit parallel arithmetic using 64-bit GPR masks.
- [x] **Cache-Oblivious i-k-j Tiling**: CPU compute loops rearranged for L1/L2 spatial locality.
- [x] **F16 Sum Vulkan tolerance**: atol relaxed to 0.1 — GPU path upcasts F16→F32 internally (architecturally more accurate than PyTorch's native F16 Kahan accumulation).
- [x] **AVX1 `vmaxps` & vectorized `exp`**: Replaced scalar loops for ReLU and Softmax exp on Ivy Bridge CPUs with zero-overhead `_mm256_max_ps` and a custom exact 256-bit taylor series float exponentiation that resolves Illegal Instruction issues with `_mm256_cvtps_epi32`.

---

#### 🔴 PRIORITY 1 — Fused Vulkan Elementwise (`mul`, `sub`, `div`)
*Root problem: `mul`/`sub`/`div` call `execute_activation_chunked` which falls back to CPU scalar. There is NO Vulkan shader for them. Benchmark ratio: `Mul f32 (vulkan)` = 2.4x, `Sub f32 (vulkan)` = 2.3x — ALL overhead is PCIe roundtrip from CPU scalar fallback.*

**How to implement — step by step:**

**Step 1: Create shader `src/shaders/elementwise.comp`**
```glsl
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) buffer BufC { float c[]; };
layout(push_constant) uniform PC {
    uint n;
    uint op; // 0=mul, 1=sub, 2=div, 3=add
} pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    if      (pc.op == 0u) c[i] = a[i] * b[i];
    else if (pc.op == 1u) c[i] = a[i] - b[i];
    else if (pc.op == 2u) c[i] = a[i] / b[i];
    else                  c[i] = a[i] + b[i];
}
```

**Step 2: Compile SPIR-V in `build.rs`**
Add a `compile_glsl("src/shaders/elementwise.comp", "elementwise.spv")` call alongside the existing shader compilations. Use the same `naga` / `glslc` pattern already used for `add.comp`, `matmul.comp` etc. The pattern in `build.rs` is:
```rust
compile_shader("src/shaders/elementwise.comp", "elementwise.spv");
```

**Step 3: Add pipeline in `backend.rs`**
In `AshBackend` struct, add:
```rust
pub pipe_elementwise: vk::Pipeline,
```
In `init_backend`, load the SPIR-V and create the pipeline using the **same descriptor set layout as `pipe_add`** (3 bindings: A in, B in, C out). This layout (`dsl_elementwise`) has bindings [0]=A, [1]=B, [2]=C, all `STORAGE_BUFFER`. The push constant block is 8 bytes: `[n: u32, op: u32]`. Use the existing `create_compute_pipeline` helper function. Also add:
```rust
pub perm_desc_elementwise: Mutex<vk::DescriptorSet>,
```

**Step 4: Create `execute_elementwise` in `backend.rs`**
Model it EXACTLY after `execute_add_into`. The only differences are:
- 3 inputs (a_raw, b_raw, res_raw) → same as add
- Push constant includes `op: u32` where mul=0, sub=1, div=2, add=3
- Use `pipe_elementwise` instead of `pipe_add`
- Use `perm_desc_elementwise` instead of `perm_desc_add`

**Step 5: Wire in `tensor.rs`**
In `Tensor::binary_op_into` (or wherever `mul`/`sub`/`div` dispatch), when `self.device != "cpu"`, call `backend::execute_elementwise(a_raw, b_raw, res_raw, op_id, dtype)` instead of the CPU scalar loop. The op_id mapping is: `"mul"→0, "sub"→1, "div"→2`.

**Success criterion**: `Mul f32 (vulkan)` ratio ≤ 1.5x PyTorch.

---

#### 🔴 PRIORITY 2 — Fused Vulkan Softmax (shared memory)
*Root problem: softmax runs 3 sequential CPU-side passes (max→exp→sum→div). Each pass copies data back through PCIe. On Bonaire with 1GB VRAM this is the single biggest PCIe bottleneck after MatMul. Benchmark: `Softmax f32 (vulkan)` = 3.31x.*

**How to implement — step by step:**

**Step 1: Create shader `src/shaders/softmax.comp`**
This is a single-pass softmax using shared memory for row-local reduction.
```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In  { float x[]; };
layout(set = 0, binding = 1) buffer       Out { float y[]; };
layout(push_constant) uniform PC { uint n; uint row_stride; } pc;
shared float sdata[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint base = row * pc.row_stride;

    // Pass 1: find row max (for numerical stability — prevents exp overflow)
    float mx = -1e38;
    for (uint i = tid; i < pc.row_stride; i += 256)
        mx = max(mx, x[base + i]);
    sdata[tid] = mx;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid+s]);
        barrier();
    }
    float row_max = sdata[0];

    // Pass 2: exp(x - max) and accumulate sum — all in shared memory, no PCIe
    float s = 0.0;
    for (uint i = tid; i < pc.row_stride; i += 256) {
        float ex = exp(x[base + i] - row_max);
        y[base + i] = ex;
        s += ex;
    }
    sdata[tid] = s;
    barrier();
    for (uint ss = 128; ss > 0; ss >>= 1) {
        if (tid < ss) sdata[tid] += sdata[tid+ss];
        barrier();
    }
    float inv_sum = 1.0 / sdata[0];

    // Pass 3: normalize
    for (uint i = tid; i < pc.row_stride; i += 256)
        y[base + i] *= inv_sum;
}
```
Dispatch: one workgroup per row. If tensor is 2D `[rows, cols]`, dispatch `(rows, 1, 1)` workgroups with `local_size_x=256`.

**Step 2: Add pipeline in `backend.rs`**
Descriptor layout: 2 bindings (In, Out). Push constant: `[n: u32, row_stride: u32]`.
Add `pub pipe_softmax: vk::Pipeline` and `pub perm_desc_softmax: Mutex<vk::DescriptorSet>`.

**Step 3: Create `execute_softmax` in `backend.rs`**
- Accepts `input_raw: &[u8]`, `rows: u32`, `cols: u32`, `res_raw: &mut [u8]`, `dtype: DataType`
- Upload via `upload_to_stage` (converts F16/BF16→F32 automatically)
- Push constant: `[rows*cols, cols]`
- Dispatch: `(rows, 1, 1)` workgroups
- Download via `download_from_stage`

**Step 4: Wire in `tensor.rs`**
Find the `softmax` method. When `self.device != "cpu"` and shape has 2 dims, call `execute_softmax`. Fall back to CPU multi-pass for 1D or when device is "cpu". The 2D case covers the entire benchmark workload (shape 2048×2048).

**Success criterion**: `Softmax f32 (vulkan)` ratio ≤ 1.5x PyTorch.

---

#### 🟡 PRIORITY 3 — Fused MatMul+Bias+ReLU Mega-Kernel (Cray-1 chaining)
*Root problem: the Sprint 2 `F.linear(x, W, b)` = `x @ W.T + b` currently requires 2 Vulkan dispatches (matmul + add) and one PCIe roundtrip for the intermediate result. For Bonaire's 1GB VRAM, materializing the full 2048×2048 F32 intermediate in VRAM costs 16MB × 2 PCIe transfers. Fusing eliminates one complete roundtrip.*

**This is Sprint 2 work but architecturally belongs here because it defines the BufC accumulation pattern.**

**How to implement — step by step:**

**Step 1: Create shader `src/shaders/matmul_fused.comp`**
Extend the existing `matmul.comp` with optional bias add and optional ReLU activation fused into the final accumulation step:
```glsl
layout(push_constant) uniform PC {
    uint M; uint N; uint K;
    uint fuse_bias;  // 1 = add bias after matmul
    uint fuse_relu;  // 1 = apply ReLU after bias
} pc;
// binding 3 = bias vector (length N), only read if fuse_bias==1
layout(set=0, binding=3) readonly buffer Bias { float bias[]; };

// At the end of the existing accumulation loop, after C[row*N+col] = acc:
if (pc.fuse_bias == 1u) acc += bias[col];
if (pc.fuse_relu == 1u) acc = max(0.0, acc);
C[row*N+col] = acc;
```

**Step 2: Add Python-facing fused op in `tensor.rs`**
Add `fn matmul_bias_relu(x, weight, bias) -> Tensor` which:
1. Calls `execute_matmul_fused(x_raw, w_raw, bias_raw, M, N, K, fuse_bias=true, fuse_relu=true, ...)`
2. Returns result tensor without any intermediate Python object allocation

**Step 3: Expose to Python as `vnn.functional.linear`**
```rust
#[pyfunction]
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> PyResult<Tensor> { ... }
```
Register in `lib.rs` as `m.add_function(wrap_pyfunction!(linear, m)?)?;`

**Success criterion**: `F.linear` single Vulkan dispatch visible in profiler. Benchmark: `MatMul+Bias f32 (vulkan)` ratio ≤ 1.2x PyTorch.

---





**Hybrid MatMul Tile-Pulling (Phase 5)**

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
- [x] `mul` — elementwise multiply (`*` operator), all dtypes, CPU+Vulkan+Hybrid
- [x] `sub` — elementwise subtract (`-` operator)
- [x] `div` — elementwise divide (`/` operator)
- [x] `scalar_mul` / `scalar_add` — broadcast scalar over tensor (e.g. `x * 2.0`)

#### Shape Manipulation
- [x] `reshape` / `view` — reinterpret shape without copy
- [x] `squeeze(dim)` / `unsqueeze(dim)` — remove/add size-1 dimensions
- [x] `flatten(start, end)` — flatten a range of dims into one

#### Activations
- [x] `gelu` — Gaussian Error Linear Unit (dominant in GPT/BERT/LLaMA)
- [x] `leaky_relu(negative_slope)` — for detection/GAN models
- [x] `elu` — Exponential Linear Unit
- [x] `tanh` — Hyperbolic Tangent
- [x] `clamp(min, max)` — clamp values

#### Reductions
- [x] `sum(dim)` — sum along axis (or whole tensor)
- [x] `mean(dim)` — mean along axis
- [x] `max(dim)` / `min(dim)` — max/min values

#### Functional Ops
- [x] `softmax(dim)` — implemented as exp → sum → div (reuses above ops)
- [x] `log_softmax(dim)` — for cross-entropy

#### Tensor Creation
- [x] `Tensor.zeros(shape, dtype, device)` — without going via numpy
- [x] `Tensor.ones(shape, dtype, device)`
- [x] `Tensor.full(shape, fill_value, dtype, device)`
- [x] `Tensor.rand(shape, dtype, device)`
- [x] `Tensor.randn(shape, dtype, device)`

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

#### io_uring DataLoader (from deep_research_on_optimization.md §4)
- [ ] **io_uring O_DIRECT SSD streaming** — replace blocking `mmap` in `unary_op_ssd` with
  `io_uring` Submission/Completion Queue pairs. Use `O_DIRECT` to bypass VFS page cache.
  Configuration: ZFS `recordsize=1M`, `TILE_SIZE` aligned to `1MB` boundaries.
  Implementation: use the `tokio-uring` or `io-uring` crate as the Rust binding.
  The rayon thread pool polls the CQ; on completion it sets tile status `EMPTY→READY_FOR_CPU`.
  *This eliminates the main page-fault stall in the 16GB SSD benchmark (currently 61s).*

---

### Sprint 3.5 — "Ultra-Legacy & 32-bit Retrofits"
*Target: Execute inference on natively 32-bit constrained hardware without AVX/SSE2 guarantee (e.g. Asus 1000HD Celeron M i686, Raspberry Pi 1 B+ ARM1176JZF-S)*

#### 32-bit Architecture Support
- [ ] `cfg(target_pointer_width = "32")` cleanups across engine internals (Usize bounds, indices).
- [ ] `Xoshiro128++` fallback for 32-bit PRNG (avoid 64-bit emulation slowdown on `Xoshiro256++`).
- [ ] strict `#UD` protection (Illegal Instruction) for AVX/SSE auto-vectorization fallbacks on non-SIMD processors.

#### extreme Memory Constraints
- [ ] Sub-256MB VRAM/RAM pooling constraints for environments like early RPi and netbooks.

---

### Sprint 4 — "Performance: Fused Kernels & AVX1"
*Target: match or beat PyTorch on modern hardware, dominate on legacy*

- [ ] **Fused MatMul+Bias+ReLU** — Vulkan mega-kernel (Cray-1 vector chaining: no PCIe for bias/act).
  *See Sprint 1.5 PRIORITY 3 for full shader + Rust implementation guide.*
- [ ] **Fused LayerNorm** — single Vulkan dispatch (mean+variance+normalize+scale+shift)
- [ ] **Fused Attention** — FlashAttention-style: tiled QKV, compute in VRAM without full materialization
- [x] **AVX1 `vmaxps` kernel for ReLU** — replace scalar loop in `avx_swar.rs`.
- [x] **AVX1 vectorized `exp`** — for softmax (polynomial approximation via `_mm256_*`)
- [ ] **Buffer pool Drop integration** — Rust `Drop` for Tensor returns Vec to BufPool automatically
- [ ] **Descriptor set caching** — permanent sets for all op types (extend current matmul/add/act)
- [ ] **Tagged-Token Dataflow** — evolve MSTS from `AtomicU32` tile counter to full TTDF tag-matching.
  Per §2.3 of `deep_research_on_optimization.md`: each `StatefulTile` gets a 64-byte-aligned
  `AtomicU32 status` field acting as MERA-400 P/Q flags. States: `EMPTY→READY_FOR_CPU→READY_FOR_GPU→GPU_COMPUTING→GPU_DONE→EMPTY`.
  Workers subscribe by CAS-looping on specific state transitions instead of a shared counter.
  This eliminates static CPU/GPU splits: the system naturally load-balances by data readiness.
- [ ] **Cooperative Matrix GLSL Shader** (`src/shaders/coop_matrix.comp`) — use `KHR_cooperative_matrix`
  extension types (`coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>`) to
  route matmul through Tensor Cores / Matrix Cores on hardware that supports them.
  The `VK_KHR_cooperative_matrix` extension is already detected in `init_backend` — only the
  shader and pipeline dispatch are missing. Guard with runtime `has_coop` feature check.
  See: NVIDIA Vulkan Cooperative Matrix blog (linked in Analiza report §KHR Cooperative Matrix).

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
