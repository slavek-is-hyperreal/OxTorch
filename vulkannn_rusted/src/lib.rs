mod backend;
mod streaming;
mod tensor;
pub use tensor::{DataType, Tensor};
pub mod buf_pool;
pub mod io_uring_engine;
pub mod crook_scheduler;
pub mod cpu;
pub mod prng;

use pyo3::prelude::*;

/// A simple dummy function to verify that Python can talk to our compiled Rust library.
#[pyfunction]
fn rust_greeting(name: &str) -> PyResult<String> {
    Ok(format!("Hello from OxTorch, {}! The Iron Age has begun.", name))
}

#[pyfunction]
fn get_available_ram_bytes() -> PyResult<usize> {
    Ok(crate::streaming::get_available_ram())
}

/// The main entry point for the `vulkannn_rusted` Python extension module.
#[pymodule]
fn vulkannn_rusted(m: &Bound<'_, PyModule>) -> PyResult<()> {
    backend::init_backend();
    streaming::init_budgets();
    streaming::init_prefetcher();
    let _ = crate::tensor::capacitor::get_capacitor(); // Eager allocation

    m.add_function(wrap_pyfunction!(rust_greeting, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_ram_bytes, m)?)?;
    m.add_class::<DataType>()?;
    m.add_class::<Tensor>()?;
    Ok(())
}
