# CPU Backend & SIMD Architecture

The CPU backend in OxTorch is designed for high performance on x86_64 and aarch64 architectures, with a specific focus on machines without AVX-512 instructions (e.g., Ivy Bridge, Haswell).

---

## 1. Folder Structure

CPU kernels are strictly organized by operation category and precision:

```text
vulkannn_rusted/src/cpu/ops/
├── binary/          # Operations on two tensors (add, sub, mul, div)
├── unary/           # Operations on one tensor (relu, gelu, exp, sigmoid)
├── matmul/          # Matrix multiplication (f32, f16, bf16, i8)
├── indexing/        # index_select, embedding
├── reduction/       # sum, mean, max, softmax
├── sequence/        # cat, stack, split, chunk
└── norm/            # layer_norm, rms_norm
```

Each operation (e.g., `relu`) has its own folder, containing files for specific data types: `f32.rs`, `f16.rs`, `bf16.rs`, `i8.rs`.

---

## 2. Manual S.O.P: Adding a New Function

To add a new CPU operation (e.g., `abs`), follow the procedure below:

### Step 1: Create the Structure
Create the folder `src/cpu/ops/unary/abs/` and add the files `mod.rs`, `f32.rs`, `f16.rs`, `bf16.rs`, `i8.rs`. Register the module in `src/cpu/ops/unary/mod.rs`.

### Step 2: Implement Kernels (SIMD -> Fallback)
In each file, implement a function with explicit specialization. Example for `f32.rs`:
- `abs_f32_avx()` – using `_mm256_andnot_ps` (sign bit masking).
- `abs_f32_sse()` – for older processors.
- `abs_f32_neon()` – for ARM.
- `abs_f32_scalar()` – mandatory fallback for all other architectures.

### Step 3: Integrate with Python API (Detaching the Fallback)
1. Add the `execute_abs` method to `vulkannn_rusted/src/tensor/ops.rs`.
2. In `vulkannn_rusted/oxtorch/tensor.py`, add the method:
   ```python
   def abs(self):
       return self._vnn.execute_abs()
   ```
Adding this method to the `Tensor` class in Python automatically "overrides" the `__getattr__` mechanism, thereby detaching the slow PyTorch fallback.

### Step 4: Testing and Parity
Create a benchmark in `tests/benchmarks/f32/abs_cpu.py` (inheriting from `BenchmarkBase`). Run it to verify "bit-perfect" parity with PyTorch.

### Step 5: Full Regression
**VERY IMPORTANT**: Before submitting a PR, run **ALL** benchmarks:
```bash
python tests/run_all_benchmarks.py
```
Ensure the new implementation does not affect `TensorPool` allocation stability or break MSTS orchestration.

---

## 3. Interaction with MSTS (Multi-threading)

CPU kernels in OxTorch **should not** use multi-threading themselves (e.g., internal `par_iter`). They are orchestrated by the MSTS system at a higher level:

*   **Path A (Direct)**: The kernel is called once on the entire buffer. It uses 1 core (maximum cache control).
*   **Path B (Single-mode)**: The kernel processes 1MB tiles sequentially. Data fits perfectly in L2, ensuring zero cache misses.
*   **Path C (Full Parallel)**: `MSTS` uses `Rayon` to dispatch different tiles to different cores. Each core executes a **single-threaded** kernel on its tile. This avoids cache contention between threads (False Sharing).

---

## 4. SIMD Implementation Rules

1. **Alignment**: Always assume data might not be 32-byte aligned (use `_mm256_loadu_ps` instead of `load_ps`).
2. **Tails**: If the tensor size is not a multiple of the register width (e.g., 8 for AVX), always handle the "tail" with a scalar loop or masked store (`AVX-512`).
3. **No Exceptions**: Kernel code must be `panic-free`.
