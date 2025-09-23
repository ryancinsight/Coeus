use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, Bound, PyResult};

mod fft;
mod functional;
mod hub;
mod nn;
mod optim;
mod schedulers;
mod tensor;
mod tokenizer;
mod utils;

// Import optimizers from optim module
use crate::optim::{Adagrad, Adam, AdamW, Sgd};
// Import neural network modules
use crate::nn::{Linear, ReLU};

#[cfg(test)]
mod tests;

// Neural network modules are not yet implemented in PyCoeus

// Reduction enum is available through the nn module

// Import available optimizers from optim crate
// Import available schedulers from schedulers crate
// Note: All schedulers temporarily disabled due to trait bound issues
// use crate::schedulers::{
//     CosineAnnealingWarmRestarts,
//     // CosineAnnealingLR, CyclicLR, ExponentialLR, LambdaLR, MultiplicativeLR, OneCycleLR, PolynomialLR, ReduceLROnPlateau, StepLR,
// };
use crate::tensor::{Device, PyTensor};
use crate::tokenizer::{BERTTokenizer, BpeTokenizer, CLIPTokenizer, Encoding, GPT2Tokenizer};
// use crate::utils::{set_num_threads, get_num_threads, manual_seed, cuda_is_available};  // Temporarily disabled
use crate::fft::{FFT, IFFT};
use crate::hub::{HubManager, ModelInfo};

/// Test function to verify PyO3 is working
#[pyfunction]
fn test_function() -> String {
    "PyCoeus test function working!".to_string()
}

/// Set the number of threads for CPU operations
#[pyfunction]
fn set_num_threads(num_threads: usize) -> PyResult<()> {
    crate::tensor::PyTensor::set_num_threads(num_threads)
}

/// Get the current number of threads for CPU operations
#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    crate::tensor::PyTensor::get_num_threads()
}

/// Set the random seed for reproducible results
#[pyfunction]
fn manual_seed(seed: u64) -> PyResult<()> {
    crate::tensor::PyTensor::manual_seed(seed)
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_is_available() -> PyResult<bool> {
    crate::tensor::PyTensor::cuda_is_available()
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

    // Optimizers - actually implemented ones only
    m.add_class::<Sgd>()?;
    m.add_class::<Adam>()?;
    m.add_class::<AdamW>()?;
    m.add_class::<Adagrad>()?;

    // Learning Rate Schedulers - working ones only
    // m.add_class::<CosineAnnealingWarmRestarts>()?; // Temporarily disabled

    // Functional API - PyTorch-compatible torch.nn.functional
    m.add_function(wrap_pyfunction!(crate::functional::linear, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::relu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::elu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::celu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::selu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::hardshrink, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::hardtanh, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::prelu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::rrelu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::tanhshrink, m)?)?;
    m.add_function(wrap_pyfunction!(crate::functional::threshold, m)?)?;

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

    // Utils - Data Loading and Processing
    m.add_class::<crate::utils::PyDataset>()?;
    m.add_class::<crate::utils::PyDataLoader>()?;
    m.add_class::<crate::utils::PyDataLoaderIter>()?;
    m.add_class::<crate::utils::PyTensorDataset>()?;
    m.add_class::<crate::utils::PyConcatDataset>()?;
    m.add_class::<crate::utils::PySubset>()?;

    // Transforms
    m.add_class::<crate::utils::PyTransform>()?;
    m.add_class::<crate::utils::PyCompose>()?;
    m.add_class::<crate::utils::PyNormalize>()?;
    m.add_class::<crate::utils::PyToTensor>()?;
    m.add_class::<crate::utils::PyRandomHorizontalFlip>()?;
    m.add_class::<crate::utils::PyRandomVerticalFlip>()?;
    m.add_class::<crate::utils::PyColorJitter>()?;

    // Metrics functions
    m.add_function(wrap_pyfunction!(crate::utils::py_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::py_top_k_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::py_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::py_classification_report, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::py_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::py_auc_roc, m)?)?;

    // Legacy utility functions
    m.add_function(wrap_pyfunction!(crate::utils::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::manual_seed, m)?)?;
    m.add_function(wrap_pyfunction!(crate::utils::cuda_is_available, m)?)?;

    Ok(())
}
