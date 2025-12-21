use super::tensor::PyTensor;
use backend::CpuBackend;
use dtype::float::Float32;
use nn::Module;
use storage::DenseStorage;
use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult, PyErr};

/// Linear function
#[pyfunction]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    // Use the functional API from nn
    let result = nn::functional::linear(&input.inner, &weight.inner, bias.map(|b| &b.inner))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// ReLU activation function
#[pyfunction]
pub fn relu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::relu(&input.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ReLU operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Sigmoid activation function
#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::sigmoid(&input.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sigmoid operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Tanh activation function
#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::tanh(&input.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tanh operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// GELU activation function
#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::gelu(&input.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// SiLU (Swish) activation function
#[pyfunction]
pub fn silu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::silu(&input.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Leaky ReLU activation function
#[pyfunction]
#[pyo3(signature = (input, negative_slope=0.01))]
pub fn leaky_relu(input: &PyTensor, negative_slope: f64) -> PyResult<PyTensor> {
    let result =
        nn::functional::leaky_relu(&input.inner, Some(negative_slope)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// ELU activation function
#[pyfunction]
#[pyo3(signature = (input, alpha=1.0))]
pub fn elu(input: &PyTensor, alpha: f64) -> PyResult<PyTensor> {
    let result = nn::functional::elu(&input.inner, Some(alpha)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Mean Squared Error loss
#[pyfunction]
pub fn mse_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::mse_loss(&input.inner, &target.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Cross-entropy loss
#[pyfunction]
pub fn cross_entropy(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::cross_entropy(&input.inner, &target.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Softmax function
#[pyfunction]
pub fn softmax(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::softmax(&input.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Functional operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Max pooling 2D
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None))]
pub fn max_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let result = nn::functional::max_pool2d(&input.inner, kernel_size, stride, padding)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("MaxPool2D failed: {:?}", e))
        })?;
    Ok(PyTensor { inner: result })
}

/// Average pooling 2D
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None))]
pub fn avg_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let result = nn::functional::avg_pool2d(&input.inner, kernel_size, stride, padding)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("AvgPool2D failed: {:?}", e))
        })?;
    Ok(PyTensor { inner: result })
}

/// Dropout function
#[pyfunction]
#[pyo3(signature = (input, p=0.5, training=true))]
pub fn dropout(input: &PyTensor, p: f64, training: bool) -> PyResult<PyTensor> {
    let mut dropout_layer = nn::Dropout::new(p);
    dropout_layer.training = training;
    let result = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(&dropout_layer, &input.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dropout operation failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Layer normalization function
#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, bias=None, eps=1e-5))]
pub fn layer_norm(
    input: &PyTensor,
    normalized_shape: Vec<usize>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: Option<f64>,
) -> PyResult<PyTensor> {
    let result = nn::functional::layer_norm(
        &input.inner,
        &normalized_shape,
        weight.map(|w| &w.inner),
        bias.map(|b| &b.inner),
        eps,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LayerNorm failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}

/// Binary cross-entropy with logits loss
#[pyfunction]
pub fn bce_with_logits_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional::bce_with_logits_loss(&input.inner, &target.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("BCEWithLogitsLoss failed: {:?}", e)))?;
    Ok(PyTensor { inner: result })
}
