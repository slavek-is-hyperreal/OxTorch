mod backend;
pub mod hardware_config;
pub mod vulkan;
mod sys_info;
mod tensor;
pub use tensor::{DataType, Tensor};
pub mod buf_pool;
pub mod io_uring_engine;
pub mod crook_scheduler;
pub mod cpu_old;
pub mod cpu;
pub mod prng;
pub mod models;

use pyo3::prelude::*;

/// A simple dummy function to verify that Python can talk to our compiled Rust library.
#[pyfunction]
fn rust_greeting(name: &str) -> PyResult<String> {
    Ok(format!("Hello from OxTorch, {}! The Iron Age has begun.", name))
}

#[pyfunction]
fn get_available_ram_bytes() -> PyResult<usize> {
    Ok((sys_info::get_sys_info().ram_available_gb * 1024.0 * 1024.0 * 1024.0) as usize)
}

/// The main entry point for the `vulkannn_rusted` Python extension module.
#[pymodule]
fn vulkannn_rusted(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sys_info::print_sys_info();
    backend::init_backend();
    // get_capacitor() is now lazy (v4 True Adaptive); it will trigger on demand.

    m.add_function(wrap_pyfunction!(rust_greeting, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_ram_bytes, m)?)?;
    m.add_class::<DataType>()?;
    m.add_class::<Tensor>()?;
    m.add_class::<models::bitnet::BitNetModel>()?;
    Ok(())
}
