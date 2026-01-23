use pyo3::prelude::*;
use pyo3::{pymodule, wrap_pyfunction, Bound, PyResult, Python};

// Module declarations
pub mod error;
pub mod fft;
pub mod functional;
pub mod hub;
pub mod linalg;
pub mod nn;
pub mod optim;
pub mod signal;
pub mod sparse;
pub mod special;
pub mod tensor;
pub mod tokenizer;
pub mod transforms;
pub mod utils;

#[pyfunction]
fn test_function() -> String {
    "PyCoeus test function working!".to_string()
}

#[pyfunction]
fn grad_enabled() -> bool {
    ::tensor::tensor_core::grad_enabled()
}

#[pyfunction]
fn set_grad_enabled(enabled: bool) -> PyResult<()> {
    ::tensor::tensor_core::set_grad_enabled(enabled);
    Ok(())
}

/// Set the number of threads for CPU operations
#[pyfunction]
fn set_num_threads(_num_threads: usize) -> PyResult<()> {
    // Future enhancement: Implement when backend supports it
    Ok(())
}

/// Get the current number of threads for CPU operations
#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    // Future enhancement: Implement when backend supports it
    Ok(1)
}

/// Set the random seed for reproducible results
#[pyfunction]
fn manual_seed(_seed: u64) -> PyResult<()> {
    // Future enhancement: Implement when backend supports it
    Ok(())
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_is_available() -> PyResult<bool> {
    // Future enhancement: CUDA not yet implemented
    Ok(false)
}

/// Python bindings for Coeus neural network library
/// Provides PyTorch-compatible API with automatic differentiation
#[pymodule]
#[pyo3(name = "_coeus")]
fn _coeus(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Register exception hierarchy
    error::register_exceptions(py, m)?;

    // Global utility functions
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(manual_seed, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;

    // Submodule registration
    crate::tensor::register(py, m)?;
    crate::nn::register(py, m)?;
    crate::optim::register(py, m)?;
    crate::functional::register(py, m)?;
    crate::fft::register(py, m)?;
    crate::linalg::register(py, m)?;
    crate::signal::register(py, m)?;
    crate::sparse::register(py, m)?;
    crate::special::register(py, m)?;
    crate::transforms::register(py, m)?;
    crate::tokenizer::register(py, m)?;
    crate::utils::register(py, m)?;
    crate::hub::register(py, m)?;

    Ok(())
}
