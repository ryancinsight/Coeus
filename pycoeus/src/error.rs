//! Comprehensive Python exception handling for PyCoeus
//!
//! This module provides a complete exception hierarchy that maps Rust errors
//! to specific Python exception types, enabling better error handling in user code.
//!
//! The exception hierarchy follows PyTorch's patterns while adding Rust-specific
//! error context for improved debugging.

use pyo3::exceptions::PyException;
use pyo3::types::PyModuleMethods;
use pyo3::{create_exception, PyErr};

// Create custom Python exception types matching the Python hierarchy
// These are registered with Python and can be caught by Python code

create_exception!(
    coeus,
    CoeusError,
    PyException,
    "Base exception for all Coeus errors"
);

create_exception!(
    coeus,
    TensorError,
    CoeusError,
    "Raised when tensor operations fail"
);

create_exception!(
    coeus,
    BackendError,
    CoeusError,
    "Raised when backend operations fail"
);

create_exception!(
    coeus,
    OptimizerError,
    CoeusError,
    "Raised when optimizer operations fail"
);

create_exception!(
    coeus,
    NNError,
    CoeusError,
    "Raised when neural network operations fail"
);

create_exception!(
    coeus,
    StorageError,
    CoeusError,
    "Raised when storage operations fail"
);

create_exception!(
    coeus,
    ShapeError,
    TensorError,
    "Raised when tensor shapes are incompatible"
);

create_exception!(
    coeus,
    DeviceError,
    BackendError,
    "Raised when device operations fail"
);

/// Convert a Rust error to the most appropriate Python exception
///
/// This function analyzes the error message to determine the most specific
/// exception type to raise. It provides better error categorization than
/// generic RuntimeError.
///
/// # Pattern Matching Rules
///
/// The function uses pattern matching on error messages to determine exception types:
/// - "shape", "dimension", "size mismatch" → ShapeError
/// - "device", "cuda", "gpu", "cpu" → DeviceError
/// - "backend", "compute" → BackendError
/// - "optimizer", "learning rate", "step" → OptimizerError
/// - "storage", "memory", "allocation" → StorageError
/// - "layer", "module", "forward", "backward" → NNError
/// - "tensor" → TensorError
/// - Default → CoeusError
///
/// # Examples
///
/// ```rust
/// use pycoeus::error::convert_error;
///
/// // Shape error
/// let err = convert_error("shape mismatch: expected [2, 3], got [3, 2]");
/// // Returns ShapeError
///
/// // Device error
/// let err = convert_error("CUDA device not available");
/// // Returns DeviceError
///
/// // Generic error
/// let err = convert_error("unknown error occurred");
/// // Returns CoeusError
/// ```
pub fn convert_error<E: std::fmt::Display>(error: E) -> PyErr {
    let err_str = error.to_string().to_lowercase();

    // Pattern match on error message to determine exception type
    // Order matters: check most specific patterns first

    // Shape-related errors (most specific tensor error)
    if err_str.contains("shape")
        || err_str.contains("dimension")
        || err_str.contains("size mismatch")
        || err_str.contains("incompatible shapes")
        || err_str.contains("broadcast")
    {
        return PyErr::new::<ShapeError, _>(error.to_string());
    }

    // Device-related errors (most specific backend error)
    if err_str.contains("device")
        || err_str.contains("cuda")
        || err_str.contains("gpu")
        || err_str.contains("cpu")
        || err_str.contains("tpu")
        || err_str.contains("npu")
    {
        return PyErr::new::<DeviceError, _>(error.to_string());
    }

    // Backend errors (general compute errors)
    if err_str.contains("backend") || err_str.contains("compute") || err_str.contains("execution") {
        return PyErr::new::<BackendError, _>(error.to_string());
    }

    // Optimizer errors
    if err_str.contains("optimizer")
        || err_str.contains("learning rate")
        || err_str.contains("step")
        || err_str.contains("gradient")
        || err_str.contains("parameter")
    {
        return PyErr::new::<OptimizerError, _>(error.to_string());
    }

    // Storage errors
    if err_str.contains("storage")
        || err_str.contains("memory")
        || err_str.contains("allocation")
        || err_str.contains("sparse")
        || err_str.contains("dense")
    {
        return PyErr::new::<StorageError, _>(error.to_string());
    }

    // Neural network errors
    if err_str.contains("layer")
        || err_str.contains("module")
        || err_str.contains("forward")
        || err_str.contains("backward")
        || err_str.contains("conv")
        || err_str.contains("linear")
        || err_str.contains("activation")
    {
        return PyErr::new::<NNError, _>(error.to_string());
    }

    // General tensor errors
    if err_str.contains("tensor") {
        return PyErr::new::<TensorError, _>(error.to_string());
    }

    // Default to base CoeusError
    PyErr::new::<CoeusError, _>(error.to_string())
}

/// Convert a Rust error to a Python exception with additional context
///
/// This function is similar to `convert_error` but allows adding contextual
/// information to the error message, making debugging easier.
///
/// # Examples
///
/// ```rust
/// use pycoeus::error::convert_error_with_context;
///
/// let err = convert_error_with_context(
///     "invalid dimensions",
///     "matrix multiplication"
/// );
/// // Returns ShapeError with message: "matrix multiplication: invalid dimensions"
/// ```
pub fn convert_error_with_context<E: std::fmt::Display>(error: E, context: &str) -> PyErr {
    let full_message = format!("{}: {}", context, error);
    convert_error(full_message)
}

/// Macro for creating tensor-related errors with context
///
/// This macro simplifies error creation by automatically adding context
/// and converting to the appropriate Python exception type.
///
/// # Examples
///
/// ```rust
/// use pycoeus::tensor_error;
///
/// fn my_operation() -> PyResult<()> {
///     tensor_error!("operation failed")
/// }
/// ```
#[macro_export]
macro_rules! tensor_error {
    ($error:expr) => {
        $crate::error::convert_error($error)
    };
    ($error:expr, $context:expr) => {
        $crate::error::convert_error_with_context($error, $context)
    };
}

/// Macro for creating backend-related errors with context
#[macro_export]
macro_rules! backend_error {
    ($error:expr) => {
        $crate::error::convert_error(format!("backend: {}", $error))
    };
    ($error:expr, $context:expr) => {
        $crate::error::convert_error_with_context(format!("backend: {}", $error), $context)
    };
}

/// Macro for creating optimizer-related errors with context
#[macro_export]
macro_rules! optimizer_error {
    ($error:expr) => {
        $crate::error::convert_error(format!("optimizer: {}", $error))
    };
    ($error:expr, $context:expr) => {
        $crate::error::convert_error_with_context(format!("optimizer: {}", $error), $context)
    };
}

/// Macro for creating neural network errors with context
#[macro_export]
macro_rules! nn_error {
    ($error:expr) => {
        $crate::error::convert_error(format!("layer: {}", $error))
    };
    ($error:expr, $context:expr) => {
        $crate::error::convert_error_with_context(format!("layer: {}", $error), $context)
    };
}

/// Macro for creating storage-related errors with context
#[macro_export]
macro_rules! storage_error {
    ($error:expr) => {
        $crate::error::convert_error(format!("storage: {}", $error))
    };
    ($error:expr, $context:expr) => {
        $crate::error::convert_error_with_context(format!("storage: {}", $error), $context)
    };
}

/// Register all exception types with Python
///
/// This function must be called during module initialization to make
/// the exception types available to Python code.
///
/// # Examples
///
/// ```rust
/// use pyo3::prelude::*;
/// use pycoeus::error::register_exceptions;
///
/// #[pymodule]
/// fn my_module(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
///     register_exceptions(py, m)?;
///     Ok(())
/// }
/// ```
pub fn register_exceptions(
    py: pyo3::Python,
    m: &pyo3::Bound<pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    m.add("CoeusError", py.get_type::<CoeusError>())?;
    m.add("TensorError", py.get_type::<TensorError>())?;
    m.add("BackendError", py.get_type::<BackendError>())?;
    m.add("OptimizerError", py.get_type::<OptimizerError>())?;
    m.add("NNError", py.get_type::<NNError>())?;
    m.add("StorageError", py.get_type::<StorageError>())?;
    m.add("ShapeError", py.get_type::<ShapeError>())?;
    m.add("DeviceError", py.get_type::<DeviceError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to test error categorization without Python initialization
    fn categorize_error(err_str: &str) -> &'static str {
        let err_lower = err_str.to_lowercase();

        if err_lower.contains("shape")
            || err_lower.contains("dimension")
            || err_lower.contains("size mismatch")
            || err_lower.contains("incompatible shapes")
            || err_lower.contains("broadcast")
        {
            "ShapeError"
        } else if err_lower.contains("device")
            || err_lower.contains("cuda")
            || err_lower.contains("gpu")
            || err_lower.contains("cpu")
            || err_lower.contains("tpu")
            || err_lower.contains("npu")
        {
            "DeviceError"
        } else if err_lower.contains("backend")
            || err_lower.contains("compute")
            || err_lower.contains("execution")
        {
            "BackendError"
        } else if err_lower.contains("optimizer")
            || err_lower.contains("learning rate")
            || err_lower.contains("step")
            || err_lower.contains("gradient")
            || err_lower.contains("parameter")
        {
            "OptimizerError"
        } else if err_lower.contains("storage")
            || err_lower.contains("memory")
            || err_lower.contains("allocation")
            || err_lower.contains("sparse")
            || err_lower.contains("dense")
        {
            "StorageError"
        } else if err_lower.contains("layer")
            || err_lower.contains("module")
            || err_lower.contains("forward")
            || err_lower.contains("backward")
            || err_lower.contains("conv")
            || err_lower.contains("linear")
            || err_lower.contains("activation")
        {
            "NNError"
        } else if err_lower.contains("tensor") {
            "TensorError"
        } else {
            "CoeusError"
        }
    }

    #[test]
    fn test_shape_error_detection() {
        assert_eq!(
            categorize_error("shape mismatch: expected [2, 3], got [3, 2]"),
            "ShapeError"
        );
    }

    #[test]
    fn test_device_error_detection() {
        assert_eq!(categorize_error("CUDA device not available"), "DeviceError");
    }

    #[test]
    fn test_optimizer_error_detection() {
        assert_eq!(categorize_error("optimizer step failed"), "OptimizerError");
    }

    #[test]
    fn test_storage_error_detection() {
        assert_eq!(categorize_error("memory allocation failed"), "StorageError");
    }

    #[test]
    fn test_nn_error_detection() {
        assert_eq!(categorize_error("layer forward pass failed"), "NNError");
    }

    #[test]
    fn test_tensor_error_detection() {
        assert_eq!(categorize_error("tensor operation failed"), "TensorError");
    }

    #[test]
    fn test_broadcast_error_is_shape_error() {
        assert_eq!(
            categorize_error("broadcast error: incompatible shapes"),
            "ShapeError"
        );
    }

    #[test]
    fn test_cuda_error_is_device_error() {
        assert_eq!(categorize_error("cuda out of memory"), "DeviceError");
    }

    #[test]
    fn test_gradient_error_is_optimizer_error() {
        assert_eq!(
            categorize_error("gradient computation failed"),
            "OptimizerError"
        );
    }

    #[test]
    fn test_sparse_error_is_storage_error() {
        assert_eq!(
            categorize_error("sparse matrix conversion failed"),
            "StorageError"
        );
    }

    #[test]
    fn test_conv_error_is_nn_error() {
        assert_eq!(categorize_error("conv2d operation failed"), "NNError");
    }

    #[test]
    fn test_default_error_is_coeus_error() {
        assert_eq!(categorize_error("unknown error occurred"), "CoeusError");
    }

    #[test]
    fn test_context_formatting() {
        let context = "matrix multiplication";
        let error = "invalid input";
        let full_message = format!("{}: {}", context, error);
        assert!(full_message.contains("matrix multiplication"));
        assert!(full_message.contains("invalid input"));
    }
}
