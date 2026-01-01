//! Simple Python exception handling for PyCoeus
//!
//! Provides basic error conversion from Rust errors to Python exceptions.

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Convert a Rust error to a Python exception
pub fn to_py_error<E: std::fmt::Display>(error: E, context: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{}: {}", context, error))
}

/// Macro for creating tensor-related errors
#[macro_export]
macro_rules! tensor_error {
    ($error:expr) => {
        $crate::error::to_py_error($error, "tensor operation failed")
    };
}
