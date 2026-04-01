# Developer Tutorial: Adding a New Operation

This tutorial walks you through the process of adding a new operator (e.g., `Sin`) to OxTorch. OxTorch's architecture requires changes in both Rust (Backend/Kerns) and Python (User API).

---

## Phase 1: Rust CPU Backend

Most operations start with a high-performance CPU implementation.

1.  **Create Kernel Files**: Add a new directory in `src/cpu/ops/unary/sin/`.
    - `src/cpu/ops/unary/sin/f32.rs`: SIMD/Scalar implementation for F32.
    - `src/cpu/ops/unary/sin/f16.rs`: Implementation for F16 (using `half::f16`).
2.  **Expose in CPU Module**: Register your kernels in `src/cpu/mod.rs`.
    ```rust
    pub fn sin_f32(in_slice: &[f32], out_slice: &mut [f32]) {
        // Implementation logic
    }
    ```
3.  **Update Support Matrix**: Add your new op to `docs/support_matrix.md`.

---

## Phase 2: Vulkan GPU Backend (Optional)

If the operation is computationally expensive, add a Vulkan shader.

1.  **Create Shader**: Add `src/shaders/sin.comp`.
    ```glsl
    #version 450
    layout(local_size_x = 16, local_size_y = 16) in; // 2D Workgroups for Tiling
    
    layout(push_constant) uniform PushConstants {
        uint M, N, K;
        uint sa_row, sa_col; // Strides for Input A
        uint sb_row, sb_col; // Strides for Input B (if binary)
        uint sc_row, sc_col; // Strides for Output C
        uint offset_a, offset_b;
    } pc;

    layout(set = 0, binding = 0) readonly buffer In { float a[]; };
    layout(set = 0, binding = 1) buffer Out { float b[]; };

    void main() {
        uint r = gl_GlobalInvocationID.y;
        uint c = gl_GlobalInvocationID.x;
        if (r < pc.M && c < pc.N) {
            uint in_idx = pc.offset_a + r * pc.sa_row + c * pc.sa_col;
            uint out_idx = r * pc.sc_row + c * pc.sc_col;
            b[out_idx] = sin(a[in_idx]);
        }
    }
    ```
2.  **Integrate with Backend**: Update `src/vulkan/ops/activation.rs` (or create a new op file).
    - Use the **44-byte** `pc_data` array (11 x 4 bytes).
    - Pass `s_row` and `s_col` for both input and output tensors to ensure **2D Stride-Awareness**.
    - Dispatch using `((N + 15) / 16, (M + 15) / 16, 1)` to match the 16x16 local workgroup size.

---

## Phase 3: Tensor Logic (Dispatch)

This is where OxTorch decides whether to use CPU, GPU, or SSD mode.

1.  **Update `ops.rs`**: Add the dispatch logic to the `Tensor` struct.
    ```rust
    impl Tensor {
        pub fn sin(&self) -> PyResult<Tensor> {
            if self.device == "cpu" {
                // Call crate::cpu::sin_f32...
            } else {
                // Call crate::backend::execute_sin_into...
            }
        }
    }
    ```
2.  **Register with PyO3**: Expose the method to Python in `src/tensor/mod.rs`.
    ```rust
    #[pymethods]
    impl Tensor {
        pub fn sin(&self) -> PyResult<Tensor> {
            self.sin()
        }
    }
    ```

---

## Phase 4: Python Wrapper (OxTorch API)

Finally, make the operation accessible to Python users.

1.  **Update `oxtorch/tensor.py`**: Add the method to the `Tensor` wrapper class for better IDE support.
    ```python
    class Tensor:
        def sin(self):
            return self._tensor.sin()
    ```

---

## Phase 5: Verification

Never skip testing.

1.  **Add Benchmark**: Create `tests/benchmark_sin.py`.
    ```python
    import oxtorch
    t = oxtorch.rand((1024, 1024))
    # Benchmark against PyTorch
    # ...
    ```
2.  **Verify Parity**: Run `python3 tests/run_all_benchmarks.py`.

## Related Documentation
- [CPU Backend](cpu_backend.md): S.O.P. for SIMD kernels.
- [Vulkan Internals](vulkan_internals.md): Pipeline and descriptor management.
- [Implementation Guide](implementation_guides.md): Design templates.
