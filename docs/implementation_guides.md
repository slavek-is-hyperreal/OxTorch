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

## Sprint 2 — MSTS PyTorch Fallback (Tile-based SSD Execution)
*Goal: Execute any PyTorch operation on massive SSD tensors without OOM.*

**Step 1: Add `unary_op_msts_pytorch` to `src/tensor/msts.rs`**
```rust
pub fn unary_op_msts_pytorch(&self, py: Python, callback: PyObject) -> PyResult<Tensor> {
    let res_tensor = Self::new_ssd(&format!("{}_pt.ssd", self.name), self.shape.clone(), self.dtype)?;
    let total_bytes = (self.shape.iter().product::<usize>() * self.dtype.bytes_per_elem()) as u64;
    let scheduler = crate::crook_scheduler::CrookScheduler::new(8);
    
    let r_handle = crate::crook_scheduler::CrookScheduler::start_read_worker(scheduler.clone(), engine_in, total_bytes);
    let w_handle = crate::crook_scheduler::CrookScheduler::start_write_worker(scheduler.clone(), engine_out, total_bytes);

    for i in 0..(total_bytes / 1048576) {
        let tile = &scheduler.ring[tile_idx];
        // Wait for TILE_READY_FOR_COMPUTE via CAS...
        // 1. Create NumPy view of tile.payload via pyo3 PyArray
        // 2. call callback.call1(py, (np_array,))
        // 3. Mark tile TILE_READY_FOR_WRITE
    }
    Ok(res_tensor)
}
```

**Step 2: Update `oxtorch/tensor.py`**
```python
def msts_pytorch_apply(self, func):
    def tile_callback(np_tile):
        with torch.no_grad():
            tt = torch.from_numpy(np_tile)
            res = func(tt)
            np_tile[:] = res.numpy()[:]
    return Tensor(self._vnn.unary_op_msts_pytorch(tile_callback))
```

**Step 3: Auto-dispatch in `__getattr__`**
If `self.device == "ssd"` and op is not natively in VNN, automatically call `msts_pytorch_apply`.

**Test**: `tests/benchmarks/monster/erf_ssd_f32.py` — verify SSD ERF runs via PyTorch callback without OOM.

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
