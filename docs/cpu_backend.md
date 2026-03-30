# CPU Backend & SIMD Architecture (v3.8.1-rc)

The CPU backend in OxTorch is designed for high performance on x86_64 and aarch64 architectures, with a specific focus on machines without AVX-512 instructions (e.g., Ivy Bridge, Haswell).

---

## 1. Unified Core (New in v3.8.1)

Version 3.8.1 introduces a new `SIMD Core` implementation in `src/cpu/ops/`. This core replaces the legacy `cpu_old` kernels with a unified dispatch model that is fully compatible with **MSTS v2**.

```text
vulkannn_rusted/src/cpu/ops/
├── binary/          # Optimized Binary Ops (add, sub, mul, div, atan2)
│   ├── fp32/        # AVX2/AVX1/SSE2/NEON/Scalar specialized kernels
│   └── bf16/        # SIMD specialized kernels for BF16
├── reduction/       # sum, mean, max with f64 accumulation
└── conversion/      # Fast f16 <-> f32 conversion via F16C
```

---

## 2. SIMD Auto-Dispatch Tiers

OxTorch detect CPU features at runtime and selects the best available specialized kernel:

| Tier | Feature | Target Hardware | Notes |
|:---|:---|:---|:---|
| **Tier 1** | `AVX2` | Haswell+ | 256-bit wide, FMA support. |
| **Tier 2** | `AVX` | Ivy Bridge / Sandy Bridge | 256-bit wide, F16C support for `f16` upcast. |
| **Tier 3** | `SSE2` | Legacy x64 | 128-bit wide, reliable fallback. |
| **Tier 4** | `NEON` | Apple Silicon / Graviton | 128-bit wide ARM specialized. |
| **Fallback** | `Scalar` | Generic | Single-element loop (safe everywhere). |

---

## 3. Adding a New Function (Integrated S.O.P)

To add a new CPU operation (e.g., `abs`), follow this integrated procedure:

### Step 1: Implement the SIMD dispatcher
Create `src/cpu/ops/unary/abs/mod.rs` and the corresponding SIMD implementations.
Ensure the `Scalar` fallback is implemented last.

### Step 2: Register in MSTS
Update `vulkannn_rusted/src/tensor/msts.rs`:
- **RAM-FastPath**: Add the dispatch in `dispatch_binary_op` (approx. Line 44) or `execute_unary_op_ssd`.
- **Hybrid/SSD**: Add the loop kernel in `execute_op_unified` (approx. Line 396).

### Step 3: Numerical Parity Verification
**CRITICAL**: Reductions (like `sum`) MUST accumulate in `f64` to prevent precision loss.
Always compare against PyTorch using the atomized benchmark suite:
```bash
python3 tests/benchmarks/f32/abs_cpu.py
```

---

## 4. Interaction with MSTS (Triple-Buffering)

CPU kernels in the Unified Core are designed to be **Thread-Safe** and **Re-entrant**. They avoid internal multi-threading to allow the `CrookScheduler` to overlap their execution with I/O tasks.

*   **MSTS Threading**: Uses a dedicated Compute Worker thread that drains the `StatefulTile` ring.
*   **SIMD Saturation**: Kernels are optimized to saturate the CPU's execution units (ports) within a single thread, maximizing IPC (Instructions Per Cycle).
