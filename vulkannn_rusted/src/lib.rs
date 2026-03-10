mod backend;
mod streaming;
mod tensor;
pub mod io_uring_engine;
pub mod crook_scheduler;
pub mod avx_swar;

use pyo3::prelude::*;

/// A simple dummy function to verify that Python can talk to our compiled Rust library.
#[pyfunction]
fn rust_greeting(name: &str) -> PyResult<String> {
    Ok(format!("Hello from VulkanNN-Rusted, {}! The Iron Age has begun.", name))
}

/// The main entry point mapped directly to the `vulkannn_rusted_test` Python module.
#[pymodule]
fn vulkannn_rusted_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the wgpu engine once upon module load
    backend::init_backend();
    // Initialize memory budgets
    streaming::init_budgets();
    // Initialize background prefetching thread
    streaming::init_prefetcher();

    m.add_function(wrap_pyfunction!(rust_greeting, m)?)?;
    m.add_class::<tensor::DataType>()?;
    m.add_class::<tensor::Tensor>()?;
    Ok(())
}
