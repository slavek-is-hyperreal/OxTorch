# Implementation Guides - OxTorch

This document contains detailed, step-by-step implementation plans for core OxTorch operations. These guides were prepared by an AI agent to ensure numerical parity and performance.

---

## 🟡 PRIORITY 1 — Fused Vulkan Elementwise (`mul`, `sub`, `div`)
*Root problem: `mul`/`sub`/`div` call `execute_activation_chunked` which falls back to CPU scalar. There is NO Vulkan shader for them. Benchmark ratio: `Mul f32 (vulkan)` = 2.4x.*

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
Add: `compile_shader("src/shaders/elementwise.comp", "elementwise.spv");`

**Step 3: Add pipeline in `backend.rs`**
Add `pub pipe_elementwise: vk::Pipeline` and `pub perm_desc_elementwise: Mutex<vk::DescriptorSet>`. Use the same descriptor layout as `pipe_add`.

**Step 4: Create `execute_elementwise` in `backend.rs`**
Model after `execute_add_into`. Push constant includes `op: u32`.

**Step 5: Wire in `tensor.rs`**
In `binary_op_into`, call `backend::execute_elementwise` when device != "cpu".

---

## 🟡 PRIORITY 2 — Fused Vulkan Softmax (Shared Memory)
*Root problem: softmax runs 3 sequential CPU-side passes. Benchmark: `Softmax f32 (vulkan)` = 3.31x.*

**Step 1: Create shader `src/shaders/softmax.comp`**
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

    float mx = -1e38;
    for (uint i = tid; i < pc.row_stride; i += 256) mx = max(mx, x[base + i]);
    sdata[tid] = mx; barrier();
    for (uint s = 128; s > 0; s >>= 1) { if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid+s]); barrier(); }
    float row_max = sdata[0];

    float s = 0.0;
    for (uint i = tid; i < pc.row_stride; i += 256) {
        float ex = exp(x[base + i] - row_max);
        y[base + i] = ex; s += ex;
    }
    sdata[tid] = s; barrier();
    for (uint ss = 128; ss > 0; ss >>= 1) { if (tid < ss) sdata[tid] += sdata[tid+ss]; barrier(); }
    float inv_sum = 1.0 / sdata[0];

    for (uint i = tid; i < pc.row_stride; i += 256) y[base + i] *= inv_sum;
}
```

**Step 2: Add pipeline in `backend.rs`**
Descriptor layout: 2 bindings (In, Out). Push constant: `[n, row_stride]`.

**Step 3: Create `execute_softmax` in `backend.rs`**
Dispatch: `(rows, 1, 1)` workgroups with `local_size_x=256`.

---

## 🟡 PRIORITY 3 — Fused MatMul+Bias+ReLU Mega-Kernel
*Goal: Eliminate PCIe roundtrip for intermediate results.*

**Step 1: Create shader `src/shaders/matmul_fused.comp`**
Extend existing `matmul.comp` with:
```glsl
layout(push_constant) uniform PC {
    uint M; uint N; uint K;
    uint fuse_bias; uint fuse_relu;
} pc;
layout(set=0, binding=3) readonly buffer Bias { float bias[]; };

// Final accumulation step:
if (pc.fuse_bias == 1u) acc += bias[col];
if (pc.fuse_relu == 1u) acc = max(0.0, acc);
C[row*N+col] = acc;
```

**Step 2: Add Python-facing fused op in `tensor.rs`**
Add `fn matmul_bias_relu(x, weight, bias) -> Tensor`.

**Step 3: Expose to Python as `vnn.functional.linear`**.
