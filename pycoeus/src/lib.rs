use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::PyResult;

mod nn;
mod optim;
mod tensor;
mod tokenizer;
// mod utils;  // Temporarily disabled
mod fft;
mod hub;

// Import only what's actually implemented
use crate::nn::{
    Conv2d,
    CrossEntropyLoss,
    Gru,
    // Actually implemented layers
    Linear,
    Lstm,
    // Actually implemented loss functions
    MseLoss,
    // Base module
    NNModule,
    // Actually implemented activations
    ReLU,
    Rnn,
    GPT2,
};
use crate::optim::{Adagrad, Adam, AdamW, Lbfgs, RMSprop, Sgd};
use crate::tensor::{Device, PyTensor};
use crate::tokenizer::{BpeTokenizer, Encoding, GPT2Tokenizer};
// use crate::utils::{set_num_threads, get_num_threads, manual_seed, cuda_is_available};  // Temporarily disabled
use crate::fft::{FFT, IFFT};
use crate::hub::{HubManager, ModelInfo};

/// Test function to verify PyO3 is working
#[pyfunction]
fn test_function() -> String {
    "PyCoeus test function working!".to_string()
}

/// Create a tensor filled with zeros
#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    let data = vec![0.0; shape.iter().product()];
    PyTensor::new(data, shape)
}

/// Create a tensor filled with ones
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    let data = vec![1.0; shape.iter().product()];
    PyTensor::new(data, shape)
}

/// Create a tensor with random normal distribution
#[pyfunction]
fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    use ::rand::prelude::*;
    use rand_distr::StandardNormal;

    let mut rng = thread_rng();
    let data: Vec<f32> = (0..shape.iter().product())
        .map(|_| rng.sample(StandardNormal))
        .collect();

    PyTensor::new(data, shape)
}

/// Create a tensor with random uniform distribution [0, 1)
#[pyfunction]
fn rand_tensor(shape: Vec<usize>) -> PyResult<PyTensor> {
    use ::rand::prelude::*;

    let mut rng = thread_rng();
    let data: Vec<f32> = (0..shape.iter().product())
        .map(|_| rng.gen::<f32>())
        .collect();

    PyTensor::new(data, shape)
}

/// Create a tensor with evenly spaced values
#[pyfunction]
fn arange(start: f32, end: f32, step: f32) -> PyResult<PyTensor> {
    let mut data = Vec::new();
    let mut current = start;

    while current < end {
        data.push(current);
        current += step;
    }

    let shape = vec![data.len()];
    PyTensor::new(data, shape)
}

/// Create an identity matrix
#[pyfunction]
fn eye(n: usize, m: usize) -> PyResult<PyTensor> {
    let mut data = vec![0.0; n * m];
    let min_dim = n.min(m);

    for i in 0..min_dim {
        data[i * m + i] = 1.0;
    }

    PyTensor::new(data, vec![n, m])
}

/// Set the number of threads for CPU operations
#[pyfunction]
fn set_num_threads(_num_threads: usize) -> PyResult<()> {
    // Placeholder implementation
    Ok(())
}

/// Get the current number of threads for CPU operations
#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    // Placeholder implementation
    Ok(1)
}

/// Set the random seed for reproducible results
#[pyfunction]
fn manual_seed(_seed: u64) -> PyResult<()> {
    // Placeholder implementation
    Ok(())
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_is_available() -> PyResult<bool> {
    // Placeholder implementation
    Ok(false)
}

/// Python bindings for Coeus neural network library
/// Provides PyTorch-compatible API with automatic differentiation
#[pymodule]
fn _core(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Test function
    m.add_function(wrap_pyfunction_bound!(test_function, py)?)?;

    // Core tensor operations
    m.add_class::<PyTensor>()?;
    m.add_class::<Device>()?;

    // Tensor creation functions
    m.add_function(wrap_pyfunction_bound!(zeros, py)?)?;
    m.add_function(wrap_pyfunction_bound!(ones, py)?)?;
    m.add_function(wrap_pyfunction_bound!(randn, py)?)?;
    m.add_function(wrap_pyfunction_bound!(rand_tensor, py)?)?;
    m.add_function(wrap_pyfunction_bound!(arange, py)?)?;
    m.add_function(wrap_pyfunction_bound!(eye, py)?)?;

    // Neural Network Layers (only what's actually implemented)
    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<Rnn>()?;
    m.add_class::<Lstm>()?;
    m.add_class::<Gru>()?;

    // Activation Functions (only what's actually implemented)
    m.add_class::<ReLU>()?;

    // Loss Functions (only what's actually implemented)
    m.add_class::<MseLoss>()?;
    m.add_class::<CrossEntropyLoss>()?;

    // Base Module
    m.add_class::<NNModule>()?;

    // Models (only what's actually implemented)
    m.add_class::<GPT2>()?;

    // Optimizers (only what's actually implemented)
    m.add_class::<Sgd>()?;
    m.add_class::<Adam>()?;
    m.add_class::<AdamW>()?;
    m.add_class::<RMSprop>()?;
    m.add_class::<Adagrad>()?;
    m.add_class::<Lbfgs>()?;

    // Tokenizers
    m.add_class::<Encoding>()?;
    m.add_class::<BpeTokenizer>()?;
    m.add_class::<GPT2Tokenizer>()?;

    // Utilities
    m.add_function(wrap_pyfunction_bound!(set_num_threads, py)?)?;
    m.add_function(wrap_pyfunction_bound!(get_num_threads, py)?)?;
    m.add_function(wrap_pyfunction_bound!(manual_seed, py)?)?;
    m.add_function(wrap_pyfunction_bound!(cuda_is_available, py)?)?;

    // FFT operations
    m.add_class::<FFT>()?;
    m.add_class::<IFFT>()?;

    // Hub operations
    m.add_class::<HubManager>()?;
    m.add_class::<ModelInfo>()?;

    Ok(())
}
