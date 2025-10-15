use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, Bound, PyResult};
use std::ops::{Add, Mul};

// Module declarations
pub mod optim;
pub mod nn;
pub mod tensor;
pub mod functional;
pub mod tokenizer;
pub mod fft;
pub mod hub;
pub mod utils;
pub mod schedulers;

// Import optimizers from optim module
use crate::optim::{Adagrad, Adam, AdamW, Sgd};
// Import neural network modules
use crate::nn::{PyLinear as Linear, PyReLU as ReLU};
// Import tensor types
use crate::tensor::{PyTensor, Device};
// Import tokenizers
use crate::tokenizer::{BERTTokenizer, BpeTokenizer, CLIPTokenizer, Encoding, GPT2Tokenizer};
// Import FFT operations
use crate::fft::{FFT, IFFT};
// Import hub operations
use crate::hub::{HubManager, ModelInfo};

// Tests are in tests/python_integration.rs

// Neural network modules are not yet implemented in PyCoeus

// Reduction enum is available through the nn module

// Import available optimizers from optim crate
// Import available schedulers from schedulers crate
// Note: All schedulers temporarily disabled due to trait bound issues
// use crate::schedulers::{
//     CosineAnnealingWarmRestarts,
//     // CosineAnnealingLR, CyclicLR, ExponentialLR, LambdaLR, MultiplicativeLR, OneCycleLR, PolynomialLR, ReduceLROnPlateau, StepLR,
// };
// use crate::utils::{set_num_threads, get_num_threads, manual_seed, cuda_is_available};  // Temporarily disabled

/// Test function to verify PyO3 is working
#[pyfunction]
fn test_function() -> String {
    "PyCoeus test function working!".to_string()
}

/// Set the number of threads for CPU operations
#[pyfunction]
fn set_num_threads(_num_threads: usize) -> PyResult<()> {
    // TODO: Implement when backend supports it
    Ok(())
}

/// Get the current number of threads for CPU operations
#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    // TODO: Implement when backend supports it
    Ok(1)
}

/// Set the random seed for reproducible results
#[pyfunction]
fn manual_seed(_seed: u64) -> PyResult<()> {
    // TODO: Implement when backend supports it
    Ok(())
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_is_available() -> PyResult<bool> {
    // TODO: CUDA not yet implemented
    Ok(false)
}

// Tensor creation functions now delegate to Rust crates
// These functions are removed and will be handled by PyTensor static methods
// that delegate to coeus_tensor::Tensor and coeus_utils::random functions

// Utility functions delegate to PyTensor static methods

/// Python bindings for Coeus neural network library
/// Provides PyTorch-compatible API with automatic differentiation
#[pymodule]
fn _core(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Test function
    m.add_function(wrap_pyfunction!(test_function, py)?)?;

    // Core tensor operations
    m.add_class::<PyTensor>()?;
    m.add_class::<Device>()?;

    // Tensor creation functions now handled by PyTensor static methods

    // Neural Network Layers
    m.add_class::<Linear>()?;
    m.add_class::<ReLU>()?;
    // Conv1dPy and ConvTranspose1dPy not yet implemented

    // Optimizers - actually implemented ones only
    m.add_class::<Sgd>()?;
    m.add_class::<Adam>()?;
    m.add_class::<AdamW>()?;
    m.add_class::<Adagrad>()?;

    // Learning Rate Schedulers - working ones only
    // m.add_class::<CosineAnnealingWarmRestarts>()?; // Temporarily disabled

    // TODO: Functional API - PyTorch-compatible torch.nn.functional (partially implemented)
    m.add_function(wrap_pyfunction!(crate::functional::linear, m)?)?;
    // TODO: Add other functional operations when implemented

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

    // Hub operations
    m.add_class::<HubManager>()?;
    m.add_class::<ModelInfo>()?;

    // TODO: Utils - Data Loading and Processing (not yet implemented)
    // m.add_class::<crate::utils::PyDataset>()?;
    // m.add_class::<crate::utils::PyDataLoader>()?;
    // m.add_class::<crate::utils::PyDataLoaderIter>()?;
    // m.add_class::<crate::utils::PyTensorDataset>()?;
    // m.add_class::<crate::utils::PyConcatDataset>()?;
    // m.add_class::<crate::utils::PySubset>()?;

    // TODO: Transforms (not yet implemented)
    // m.add_class::<crate::utils::PyTransform>()?;
    // m.add_class::<crate::utils::PyCompose>()?;
    // m.add_class::<crate::utils::PyNormalize>()?;
    // m.add_class::<crate::utils::PyToTensor>()?;
    // m.add_class::<crate::utils::PyRandomHorizontalFlip>()?;
    // m.add_class::<crate::utils::PyRandomVerticalFlip>()?;
    // m.add_class::<crate::utils::PyColorJitter>()?;

    // TODO: Metrics functions (not yet implemented)
    // m.add_function(wrap_pyfunction!(crate::utils::py_accuracy, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_top_k_accuracy, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_confusion_matrix, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_classification_report, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_mean_squared_error, m)?)?;
    // m.add_function(wrap_pyfunction!(crate::utils::py_auc_roc, m)?)?;

    // Remove redundant utilities - already added above

    Ok(())
}

use coeus_tensor::{Tensor, CpuBackend};
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

type CpuTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;
