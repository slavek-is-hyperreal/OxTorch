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

/// Override the Tier III parallelism threshold for one op×dtype at runtime.
/// `op` is the threshold name (e.g. "ADD_F32"); pass "ALL" to set every threshold.
/// Lets Python-side benchmarking sweep thresholds without recompiling.
#[pyfunction]
fn set_par_threshold(op: &str, n: usize) -> PyResult<bool> {
    if op.eq_ignore_ascii_case("ALL") {
        cpu::thresholds::set_all(n);
        return Ok(true);
    }
    match cpu::thresholds::Threshold::from_name(op) {
        Some(t) => { cpu::thresholds::set(t, n); Ok(true) }
        None => Ok(false),
    }
}

/// Test hook: run a CPU f32 unary op over a flat array and return the result.
/// Exists so the Python parity suite can reach the RAM-path transcendental
/// kernels directly (the high-level unary path is not exposed to Python for
/// in-memory tensors). Not a production API.
#[pyfunction]
fn cpu_unary_f32(op: &str, x: Vec<f32>) -> PyResult<Vec<f32>> {
    let mut out = vec![0f32; x.len()];
    match op {
        "exp" => cpu::exp_f32(&x, &mut out),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("cpu_unary_f32: unknown op {op}"))),
    }
    Ok(out)
}

/// Pin the dispatch ladder to a specific SIMD tier ("scalar"/"sse2"/"avx1"/
/// "avx2"/"neon"/"swar"), or "auto"/"" to restore normal detection. Requests
/// wider than the CPU supports (or off-family) are ignored, not executed.
/// Debugging + benchmarking aid: lets you measure a lower tier on a higher CPU.
#[pyfunction]
fn force_arch(name: &str) -> PyResult<bool> {
    if name.is_empty() || name.eq_ignore_ascii_case("auto") {
        cpu::dispatch::force_arch(None);
        return Ok(true);
    }
    Ok(cpu::dispatch::force_arch_by_name(name))
}

/// The main entry point for the `vulkannn_rusted` Python extension module.
#[pymodule]
fn vulkannn_rusted(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sys_info::print_sys_info();
    backend::init_backend();
    // get_capacitor() is now lazy (v4 True Adaptive); it will trigger on demand.

    m.add_function(wrap_pyfunction!(rust_greeting, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_ram_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(set_par_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(force_arch, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_unary_f32, m)?)?;
    m.add_class::<DataType>()?;
    m.add_class::<Tensor>()?;
    m.add_class::<models::bitnet::BitNetModel>()?;
    Ok(())
}
