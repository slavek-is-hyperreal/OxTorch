mod backend;
mod streaming;
mod tensor;
pub use tensor::{DataType, Tensor};
pub mod buf_pool;
pub mod io_uring_engine;
pub mod crook_scheduler;
pub mod cpu;
pub mod swar_int8;
pub mod tiling_cpu;
pub mod prng;

use pyo3::prelude::*;

/// A simple dummy function to verify that Python can talk to our compiled Rust library.
#[pyfunction]
fn rust_greeting(name: &str) -> PyResult<String> {
    Ok(format!("Hello from VulkanNN-Rusted, {}! The Iron Age has begun.", name))
}

/// The main entry point mapped directly to the `vulkannn_rusted_test` Python module.
#[pymodule]
fn vulkannn_rusted(m: &Bound<'_, PyModule>) -> PyResult<()> {
    backend::init_backend();
    streaming::init_budgets();
    streaming::init_prefetcher();

    m.add_function(wrap_pyfunction!(rust_greeting, m)?)?;
    m.add_class::<DataType>()?;
    m.add_class::<Tensor>()?;
    Ok(())
}
