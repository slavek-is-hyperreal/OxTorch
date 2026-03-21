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

## ✅ DONE (Sprint 4, pulled forward) — MSTS Dual-Path Dispatch + Compile-Time Burn-In

**Problem:** `unary_op_ssd` currently always spawns 2 threads + 8MB ring regardless of tensor size.
For tensors < ~32 MB this overhead dominates compute time.

**Architecture — 3 paths burned in at compile time:**

```
                  MSTS_DIRECT_MAX (e.g. 4 MB)
                  ↑
tensor_bytes ─────┤──── DIRECT PATH (no threads, mmap read_exact, single AVX loop)
                  │
                  │  MSTS_SINGLE_MAX (e.g. 32 MB)
                  ↑
                  ├──── SINGLETHREAD PATH (1 io_uring read worker, small ring)
                  │     ring_size = MSTS_RING_DEPTH_SMALL (2 tiles × TILE_SMALL)
                  │
                  └──── FULL PATH (2 workers, Rayon parallel compute)
                        ring_size = MSTS_RING_DEPTH (e.g. 4 tiles × 4 MB)
```

**Step 1: `build.rs` — emit 5 compile-time constants**

```rust
fn main() {
    // Set by build server (binary_distribution.md), or derived from sysfs locally.
    // See: docs/binary_distribution.md for the full formula table.
    let l2_kb: usize = read_l2_kb().unwrap_or(256);           // per-core L2
    let l3_mb: usize = read_l3_mb().unwrap_or(6);             // shared L3

    // Direct path: tensor fits in L3 → pure mmap, no ring
    let direct_max   = (l3_mb * 1024 * 1024) / 2;             // 50% of L3
    // Single-thread path: tile fits nicely in L2
    let tile_small   = (l2_kb * 1024 * 3) / 4;                // 75% of L2
    let ring_small   = 2usize;                                 // read-ahead 1 tile
    // Full path: tile for SATA sequential throughput burst
    let tile_large   = 4 * 1024 * 1024usize;                   // 4 MB
    let ring_large   = std::cmp::min(l3_mb / 4, 8).max(2);    // up to L3/tile_large

    println!("cargo:rustc-env=MSTS_DIRECT_MAX={}", direct_max);
    println!("cargo:rustc-env=MSTS_TILE_SMALL={}", tile_small);
    println!("cargo:rustc-env=MSTS_RING_SMALL={}", ring_small);
    println!("cargo:rustc-env=MSTS_TILE_BYTES={}", tile_large);   // existing var
    println!("cargo:rustc-env=MSTS_RING_DEPTH={}", ring_large);   // existing var
}
```

For i5-3450 (L2=256KB, L3=6MB) this computes:
- `MSTS_DIRECT_MAX` = 3 MB (tensor < 3 MB → zero thread overhead)
- `MSTS_TILE_SMALL` = 192 KB (fits in L2, hot in cache for AVX)
- `MSTS_RING_SMALL` = 2
- `MSTS_TILE_BYTES` = 4 MB
- `MSTS_RING_DEPTH` = min(6/4, 8) = 2

**Step 2: Constants in `msts.rs`**

```rust
const DIRECT_MAX:  usize = const_parse_env!("MSTS_DIRECT_MAX",  3_145_728);
const TILE_SMALL:  usize = const_parse_env!("MSTS_TILE_SMALL",    196_608);
const RING_SMALL:  usize = const_parse_env!("MSTS_RING_SMALL",          2);
const TILE_LARGE:  usize = const_parse_env!("MSTS_TILE_BYTES",  4_194_304);
const RING_LARGE:  usize = const_parse_env!("MSTS_RING_DEPTH",          4);
```

Use a `macro_rules! const_parse_env` helper since `env!()` returns `&str`, not integer.

**Step 3: Dispatch in `unary_op_ssd`**

```rust
pub fn unary_op_ssd(&self, op: &str, p1: f32, p2: f32) -> PyResult<Tensor> {
    let total_bytes = self.total_bytes();
    if total_bytes <= DIRECT_MAX {
        return self.unary_op_ssd_direct(op, p1, p2);        // Path A
    } else if total_bytes <= 32 * 1024 * 1024 {
        return self.unary_op_ssd_single(op, p1, p2,         // Path B
            TILE_SMALL, RING_SMALL);
    } else {
        return self.unary_op_ssd_full(op, p1, p2,           // Path C
            TILE_LARGE, RING_LARGE);
    }
}
```

**Path A — `unary_op_ssd_direct`:**
- Read the entire mmap via `engine.read_exact(0, total_bytes, &mut buf)`
- Single-thread AVX2 compute loop
- Write result back via `engine_out.write_all()`
- No thread spawn, no atomics, no spin-loop

**Path B — `unary_op_ssd_single`:**
- `CrookScheduler::new(RING_SMALL)` with `TILE_SMALL`-sized tiles
- Only 1 background thread (IO read worker)
- Compute inline on main thread (no Rayon) → stays in L2

**Path C — `unary_op_ssd_full` (current code, refactored):**
- `CrookScheduler::new(RING_LARGE)` with `TILE_LARGE`-sized tiles
- 2 background threads (read + write)
- `rayon` parallel compute per tile

**Integration with `binary_distribution.md`:**

The build server (`docs/binary_distribution.md` Sprint 6) already passes `MSTS_TILE_BYTES`
and `MSTS_RING_DEPTH` as env vars. This plan **extends** that by adding 3 new env vars.
The build server reads the same sysfs sources and emits all 5 at compile time.
Result: each `.whl` file has thresholds burned in for the exact CPU of the target machine.

---



## ✅ DONE (Sprint 4, pulled forward) — TensorPool Slab Allocator

**Problem solved:** Every binary op (`sub`, `mul`, `div`) was allocating a fresh `Vec<u8>` output buffer, causing 2–6× regression vs PyTorch for small-to-medium tensors.

**Implemented in `src/tensor/pool.rs`:**
- 6-bucket thread-local pool: 4KB, 64KB, 1MB, 4MB, 16MB, 64MB
- Allocates as `Vec<f32>` → guarantees 4-byte alignment (safe for `bytemuck` and `io_uring`)
- `Storage::drop` returns buffer via manual `Vec::from_raw_parts` reconstruction (avoids `bytemuck::cast_vec` AlignmentMismatch)
- Wired into `linalg.rs` via `TENSOR_POOL.with(|pool| pool.borrow_mut().alloc(n))` on all output paths

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
