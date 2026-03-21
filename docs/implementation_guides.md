# Implementation Guides — OxTorch

Detailed step-by-step guides for core OxTorch operations. Items marked ✅ are complete.

---

## ✅ DONE — Fused Vulkan Elementwise (`mul`, `sub`, `div`, `add`)

Implemented as `src/shaders/elementwise.comp.spv`. Descriptor layout: 3 bindings (A, B, C). Push constant: `[n, op]`.
Pipeline: `pipe_elementwise` in `AshBackend`. Function: `execute_elementwise` in `backend.rs`.

---

## ✅ DONE — Vulkan Softmax

Implemented as `src/shaders/softmax.wgsl.spv`. Pipeline: `pipe_softmax` in `AshBackend`.
Uses `pipe_layout_act` (same layout as activations — 1 input, 1 output binding).

---

## ✅ DONE — BitNet 1.58b Linear

BitLinear implemented in both CPU (`src/cpu/ops/bit_linear.rs`) and Vulkan (`src/shaders/bit_linear.comp.spv`).
Accepts Ternary weight tensor + Int8 activation. Zero-multiplication ternary accumulation.

---

## ✅ DONE — MSTS PyTorch Fallback (Tile-based SSD Execution)
*Goal: Execute any PyTorch operation on massive SSD tensors without OOM.*

Implemented in `src/tensor/msts.rs` (`unary_op_msts_pytorch`) and `oxtorch/tensor.py` (`msts_pytorch_apply`).
Automatically handles non-native operations on SSD tensors by streaming 1MB tiles through PyTorch.

### How to swap PyTorch Fallback for Native MSTS
When you want to implement a native optimized kernel for an operation that is currently using the PyTorch fallback (e.g. `erf`):

1. **Rust Backend**:
   - Implement the kernel in `src/tensor/msts.rs` within `unary_op_ssd` (for unary ops) or as a new specialized method.
   - Use `rayon` for parallel processing of the tile payload.
2. **Python API**:
   - Add the specific method (e.g. `def erf(self):`) to `oxtorch/tensor.py`.
   - Have it call `self._vnn.erf()` (the native Rust method).
3. **Auto-Dispatch**:
   - Because `__getattr__` only triggers if the attribute is NOT found, once you add the native `erf` method to the Python `Tensor` class, the PyTorch fallback will be automatically "disconnected" for that operation.

---

## Sprint 4 — Fused MatMul+Bias+ReLU Mega-Kernel
*Goal: Eliminate PCIe roundtrip for intermediate results.*

**Step 1: Extend `src/shaders/matmul_tiled.comp`**
```glsl
layout(push_constant) uniform PC {
    uint M; uint N; uint K;
    uint act_type;  // 0=none, 1=relu, 2=gelu
    uint has_bias;
    uint transpose_b;
} pc;
layout(set=0, binding=3) readonly buffer Bias { float bias[]; };

// At final accumulation:
if (pc.has_bias == 1u) acc += bias[col];
if (pc.act_type == 1u) acc = max(0.0, acc);
C[row*N+col] = acc;
```

`execute_matmul_with_bias` in `backend.rs` already partially implements this — wire it up fully.

---

## Sprint 4 — Descriptor Set Caching

Currently each GPU dispatch allocates and frees a descriptor set from the pool. This adds latency.
Cache sets per pipeline in `AshBackend::buffer_cache` or a dedicated `desc_cache`.

---

## Sprint 3 — Cooperative Matrix GLSL Shader

`KHR_cooperative_matrix` for Tensor Cores — check device extension support at `init_backend()`.
Fallback to current tiled shader if unsupported.

---

## Sprint 4 — TensorPool Slab Allocator

**Problem:** Every binary op (`sub`, `mul`, `div`) allocates a fresh `Vec<u8>` output buffer.
This causes 2–6× regression vs PyTorch for small-to-medium tensors (PyTorch has C++ memory pool).

**Fix:**
1. Create `TensorPool` in `src/tensor/pool.rs`:
   ```rust
   pub struct TensorPool {
       // Size classes: 4KB, 64KB, 1MB, 16MB, 256MB
       buckets: [Vec<Vec<u8>>; 5],
   }
   impl TensorPool {
       pub fn alloc(&mut self, n_bytes: usize) -> Vec<u8> { ... }
       pub fn free(&mut self, buf: Vec<u8>) { ... }
   }
   ```
2. Thread-local `TENSOR_POOL: RefCell<TensorPool>` — no locking overhead on CPU path.
3. In `linalg.rs` `execute_binary_op`: replace `vec![0u8; n_bytes]` with `pool.alloc(n_bytes)`.
4. Return buffer to pool on `Tensor` Drop.

---

## Sprint 2 Regression — `sub_i8_swar` is Scalar Stub

**Problem:** `sub_i8_swar()` in `src/cpu/ops/binary/sub/sub_i8.rs` calls `sub_i8_scalar()`.
SWAR subtraction needs borrow-bit isolation which was deferred.

**Fix (x86_64 path — simplest):**
```rust
fn sub_i8_swar(a: &[i8], b: &[i8], res: &mut [i8]) {
    // On x86_64 we always have at least SSE2; borrow-bit issue is solved by _mm_subs_epi8.
    // Just call scalar — or better: gate SWAR on aarch64 only and use SSE2 on x86_64.
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is always available on x86_64
        // _mm_subs_epi8 handles saturating sub including borrow
        return sub_i8_sse2(a, b, res);
    }
    sub_i8_scalar(a, b, res);
}
```
Add `sub_i8_sse2` using `_mm_subs_epi8` (128-bit, always available on x86_64).
This eliminates scalar loop for all x86_64 machines without AVX2.
