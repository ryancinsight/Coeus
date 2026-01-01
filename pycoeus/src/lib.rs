use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, Bound, PyErr, PyResult};
use std::vec::Vec;

// Module declarations
pub mod error; // NEW - Comprehensive Python exception handling layer
pub mod fft;
pub mod functional;
pub mod hub;
pub mod linalg;
pub mod nn;
pub mod optim;
pub mod schedulers;
pub mod sparse;
pub mod tensor;
pub mod tokenizer;
pub mod transforms; // ENABLED - Transform pipelines implemented
pub mod utils; // ENABLED - PyO3 advanced features implemented

// Import optimizers from optim module
use crate::optim::{Adagrad, Adam, AdamW, RMSprop, Sgd};
// Import neural network modules
use crate::nn::{
    PyAdaptiveAvgPool1d, PyAdaptiveAvgPool2d, PyAvgPool1d, PyAvgPool2d, PyBatchNorm2d, PyConv2D,
    PyDropout, PyEmbedding, PyGRU, PyGeLU, PyLSTM, PyLayerNorm, PyLinear, PyMaxPool1d, PyMaxPool2d,
    PyPReLU, PyRNN, PyReLU, PySequential, PySiLU, PySigmoid, PyTanh,
};
// Import tensor types
use crate::tensor::{Device, PyTensor};
// Import tokenizers
use crate::tokenizer::{BERTTokenizer, BpeTokenizer, CLIPTokenizer, Encoding, GPT2Tokenizer};
// Import FFT operations
use crate::fft::{FFT, IFFT};
// Import hub operations
use crate::hub::{HubManager, ModelInfo};
// Import utils operations - ENABLED with PyO3 trait object support
use crate::utils::{
    PyConcatDataset, PyDataLoader, PyDataLoaderIter, PySubset, PyTensorBatch, PyTensorDataset,
    PyTensorSample,
};
// Import transform operations - ENABLED
use crate::transforms::{
    compose, normalize, random_apply, resize, to_tensor, PyCompose, PyNormalize, PyRandomApply,
    PyResize, PyToTensor,
};

// Tests are in tests/python_integration.rs

// Neural network modules are not yet implemented in PyCoeus

// Reduction enum is available through the nn module

// Import available optimizers from optim crate
// Import available schedulers from schedulers crate
// Note: All schedulers temporarily disabled due to trait bound issues
use crate::schedulers::{
    CosineAnnealingLR, ExponentialLR, MultiStepLR, OneCycleLR, ReduceLROnPlateau, StepLR,
};
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

// Tensor creation functions now delegate to Rust crates
// These functions are removed and will be handled by PyTensor static methods
// that delegate to coeus_tensor::Tensor and coeus_utils::random functions

// Utility functions delegate to PyTensor static methods

/// Create a tensor filled with zeros
#[pyfunction]
fn tensor_zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::zeros(shape)
}

/// Create a tensor filled with ones
#[pyfunction]
fn tensor_ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::ones(shape)
}

/// Create a tensor with random values from a standard normal distribution
#[pyfunction]
fn tensor_randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::randn(shape)
}

/// Create a tensor with random values from a uniform distribution [0, 1)
#[pyfunction]
fn tensor_rand(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::rand(shape)
}

/// Create a tensor with random integers from [low, high)
#[pyfunction]
fn tensor_randint(low: i64, high: i64, shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::randint(low, high, shape)
}

/// Create a tensor filled with zeros with the same shape as input
#[pyfunction]
fn tensor_zeros_like(input: &PyTensor) -> PyResult<PyTensor> {
    PyTensor::zeros_like(input)
}

/// Create a tensor filled with ones with the same shape as input
#[pyfunction]
fn tensor_ones_like(input: &PyTensor) -> PyResult<PyTensor> {
    PyTensor::ones_like(input)
}

/// Create a tensor filled with a constant value with the same shape as input
#[pyfunction]
fn tensor_full_like(input: &PyTensor, fill_value: f32) -> PyResult<PyTensor> {
    PyTensor::full_like(input, fill_value)
}

/// Python bindings for Coeus neural network library
/// Provides PyTorch-compatible API with automatic differentiation
#[pymodule]
#[pyo3(name = "_coeus")]
fn _coeus(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Test function
    m.add_function(wrap_pyfunction!(test_function, py)?)?;
    m.add_function(wrap_pyfunction!(grad_enabled, py)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, py)?)?;

    // Core tensor operations
    m.add_class::<PyTensor>()?;
    m.add_class::<Device>()?;

    // Tensor creation functions now handled by PyTensor static methods
    m.add_function(wrap_pyfunction!(tensor_zeros, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ones, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_randn, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_rand, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_randint, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_zeros_like, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_ones_like, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_full_like, py)?)?;

    // Neural Network Layers - PyO3 bindings implemented with trait object support
    m.add_class::<PyLinear>()?;
    m.add_class::<PyConv2D>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PySequential>()?; // Sequential container - trait object support foundation implemented
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyRNN>()?;
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyGRU>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PyGeLU>()?;
    m.add_class::<PySiLU>()?;
    m.add_class::<PyPReLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;
    m.add_class::<PyMaxPool1d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyAvgPool1d>()?;
    m.add_class::<PyAvgPool2d>()?;
    m.add_class::<PyAdaptiveAvgPool1d>()?;
    m.add_class::<PyAdaptiveAvgPool2d>()?;

    // Optimizers - actually implemented ones only
    m.add_class::<Sgd>()?;
    m.add_class::<Adam>()?;
    m.add_class::<AdamW>()?;
    m.add_class::<Adagrad>()?;
    m.add_class::<RMSprop>()?;

    // Learning Rate Schedulers - working ones only
    // Learning Rate Schedulers
    m.add_class::<StepLR>()?;
    m.add_class::<ExponentialLR>()?;
    m.add_class::<CosineAnnealingLR>()?;
    m.add_class::<MultiStepLR>()?;
    m.add_class::<ReduceLROnPlateau>()?;
    m.add_class::<OneCycleLR>()?;

    // Functional API - PyTorch-compatible torch.nn.functional
    m.add_function(wrap_pyfunction!(crate::functional::linear, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::relu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::silu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::elu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(functional::max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(functional::avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(functional::dropout, m)?)?;
    m.add_function(wrap_pyfunction!(functional::layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(functional::bce_with_logits_loss, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;

    // Tokenizers
    m.add_class::<Encoding>()?;
    m.add_class::<BpeTokenizer>()?;
    m.add_class::<GPT2Tokenizer>()?;
    m.add_class::<CLIPTokenizer>()?;
    m.add_class::<BERTTokenizer>()?;

    // Utilities now handled by delegation to Rust crates
    m.add_function(wrap_pyfunction!(set_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(manual_seed, py)?)?;
    m.add_function(wrap_pyfunction!(cuda_is_available, py)?)?;

    // FFT operations
    m.add_class::<FFT>()?;
    m.add_class::<IFFT>()?;

    // Sparse operations
    m.add_class::<crate::sparse::PySparseCsrTensor>()?;
    m.add_class::<crate::sparse::PyCooTensor>()?;

    // Hub operations
    m.add_class::<HubManager>()?;
    m.add_class::<ModelInfo>()?;

    // Utils - Data Loading and Processing - ENABLED with PyO3 trait object support
    m.add_class::<PyTensorDataset>()?;
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyDataLoaderIter>()?;
    m.add_class::<PyTensorSample>()?;
    m.add_class::<PyTensorBatch>()?;
    m.add_class::<PyConcatDataset>()?; // Now implemented with trait object support - Sprint 37 core deliverable
    m.add_class::<PySubset>()?; // Now implemented with trait object support - Sprint 37 core deliverable

    // Transforms - Data preprocessing pipeline - ENABLED
    m.add_class::<PyToTensor>()?;
    m.add_class::<PyNormalize>()?;
    m.add_class::<PyResize>()?;
    m.add_class::<PyRandomApply>()?;
    m.add_class::<PyCompose>()?;
    // Transform factory functions
    m.add_function(wrap_pyfunction!(to_tensor, py)?)?;
    m.add_function(wrap_pyfunction!(normalize, py)?)?;
    m.add_function(wrap_pyfunction!(resize, py)?)?;
    m.add_function(wrap_pyfunction!(random_apply, py)?)?;
    m.add_function(wrap_pyfunction!(compose, py)?)?;
    // Additional transforms to be implemented:
    // m.add_class::<PyRandomHorizontalFlip>()?;
    // m.add_class::<PyRandomVerticalFlip>()?;
    // m.add_class::<PyColorJitter>()?;

    // TODO: Metrics functions (not yet implemented)
    // m.add_function(wrap_pyfunction!(crate::utils::py_accuracy, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_top_k_accuracy, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_confusion_matrix, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_classification_report, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_mean_squared_error, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_auc_roc, m)?)?;

    // Error monitoring functions
    // Note: Custom error classes removed for simplicity - using standard Python RuntimeError

    // Linalg
    m.add_function(wrap_pyfunction!(crate::linalg::inv, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::norm, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::vector_norm, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::det, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::solve, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::qr, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::svd, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn cat(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    let rust_tensors: Vec<_> = tensors.into_iter().map(|p| p.inner.clone()).collect();
    let result =
        ::tensor::ops::tensor_ops::concatenate_tensors(&rust_tensors, dim).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("cat failed: {:?}", e))
        })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn stack(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    let mut reshaped_tensors = Vec::with_capacity(tensors.len());
    for p in tensors {
        let mut shape = p.inner.shape().dims().to_vec();
        shape.insert(dim, 1);
        let reshaped = p
            .inner
            .clone()
            .reshape(&shape.iter().map(|&x| x as isize).collect::<Vec<_>>())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "stack reshape failed: {:?}",
                    e
                ))
            })?;
        reshaped_tensors.push(reshaped);
    }
    let result =
        ::tensor::ops::tensor_ops::concatenate_tensors(&reshaped_tensors, dim).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("stack cat failed: {:?}", e))
        })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmax(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.argmax(dim, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmin(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.argmin(dim, keepdim)
}
