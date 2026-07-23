//! Matrix multiplication kernels.
//!
//! MOVE-NOT-REWRITE (Wave 5): relocated verbatim from `cpu_old/ops/matmul/`.
//! Backends stay the `matrixmultiply` crate (sgemm). Zero algorithmic changes,
//! no custom GEMM. bf16/f16 tile through f32 via the same TensorPool round-trip
//! the legacy code used. Rayon parallelism (over M-blocks) lives here in the
//! dtype files exactly as legacy had it — matmul is inherently a Tier-III-only
//! op (no per-element SIMD ladder), so there is no arch/{dtype} split.

mod f32;
mod f16;
mod bf16;

pub use f32::{matmul_f32, linear_f32};
pub use f16::{matmul_f16, linear_f16};
pub use bf16::{matmul_bf16, linear_bf16};
