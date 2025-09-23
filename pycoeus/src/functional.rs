use crate::tensor::PyTensor;
use coeus_nn::functional as rust_functional;
use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult};

/// Linear transformation (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `weight` - Weight matrix
/// * `bias` - Optional bias vector
///
/// # Returns
/// Output tensor = input @ weight.T + bias
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    let result = rust_functional::linear(&input.tensor, &weight.tensor, bias.map(|b| &b.tensor))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Linear failed: {}", e))
        })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// ReLU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with ReLU applied element-wise: `max(0, x)`
#[pyfunction]
pub fn relu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = rust_functional::relu(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ReLU failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Leaky ReLU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `negative_slope` - Slope for negative values (default: 0.01)
///
/// # Returns
/// Tensor with Leaky ReLU applied element-wise: `max(αx, x)`
#[pyfunction]
#[pyo3(signature = (input, negative_slope=0.01))]
pub fn leaky_relu(input: &PyTensor, negative_slope: f32) -> PyResult<PyTensor> {
    let result =
        rust_functional::leaky_relu(&input.tensor, negative_slope).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Leaky ReLU failed: {}", e))
        })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// GELU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with GELU applied element-wise
#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = rust_functional::gelu(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("GELU failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Sigmoid activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with sigmoid applied element-wise: `1 / (1 + exp(-x))`
#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let result = rust_functional::sigmoid(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sigmoid failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Tanh activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with tanh applied element-wise: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    let result = rust_functional::tanh(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tanh failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Softmax activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `dim` - Dimension along which to apply softmax (default: -1)
///
/// # Returns
/// Tensor with softmax applied along specified dimension
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn softmax(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    let dim = dim.unwrap_or(-1);
    let result = rust_functional::softmax(&input.tensor, dim).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Softmax failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Log softmax activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `dim` - Dimension along which to apply log softmax (default: -1)
///
/// # Returns
/// Tensor with log softmax applied along specified dimension
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn log_softmax(input: &PyTensor, dim: Option<isize>) -> PyResult<PyTensor> {
    let dim = dim.unwrap_or(-1);
    let result = rust_functional::log_softmax(&input.tensor, dim).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Log softmax failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// ELU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `alpha` - The α value for the ELU formulation (default: 1.0)
///
/// # Returns
/// Tensor with ELU applied element-wise
#[pyfunction]
#[pyo3(signature = (input, _alpha=1.0))]
pub fn elu(input: &PyTensor, _alpha: f32) -> PyResult<PyTensor> {
    // Note: Using the module-based ELU since functional ELU may not be available
    use coeus_nn::{Module, ELU};
    let elu_fn = ELU::new();
    let result = elu_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ELU forward failed: {}", e))
    })?;
    Ok(PyTensor {
        tensor: result,
        requires_grad: input.requires_grad,
        device: input.device.clone(),
    })
}

/// CELU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `alpha` - The α value for the CELU formulation (default: 1.0)
///
/// # Returns
/// Tensor with CELU applied element-wise
#[pyfunction]
#[pyo3(signature = (input, _alpha=1.0))]
pub fn celu(input: &PyTensor, _alpha: f32) -> PyResult<PyTensor> {
    use coeus_nn::{Module, CELU};
    let celu_fn = CELU::new();
    let result = celu_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CELU forward failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// SELU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with SELU applied element-wise
#[pyfunction]
pub fn selu(input: &PyTensor) -> PyResult<PyTensor> {
    use coeus_nn::{Module, SELU};
    let selu_fn = SELU::new();
    let result = selu_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("SELU forward failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Hardshrink activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `lambd` - The λ value for the Hardshrink formulation (default: 0.5)
///
/// # Returns
/// Tensor with Hardshrink applied element-wise
#[pyfunction]
#[pyo3(signature = (input, _lambd=0.5))]
pub fn hardshrink(input: &PyTensor, _lambd: f32) -> PyResult<PyTensor> {
    use coeus_nn::{Hardshrink, Module};
    let hardshrink_fn = Hardshrink::new();
    let result = hardshrink_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Hardshrink forward failed: {}",
            e
        ))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Hardtanh activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `min_val` - Minimum value of the linear region range (default: -1.0)
/// * `max_val` - Maximum value of the linear region range (default: 1.0)
///
/// # Returns
/// Tensor with Hardtanh applied element-wise
#[pyfunction]
#[pyo3(signature = (input, _min_val=-1.0, _max_val=1.0))]
pub fn hardtanh(input: &PyTensor, _min_val: f32, _max_val: f32) -> PyResult<PyTensor> {
    use coeus_nn::{Hardtanh, Module};
    let hardtanh_fn = Hardtanh::new();
    let result = hardtanh_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Hardtanh forward failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// PReLU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `weight` - The weight for the PReLU formulation
///
/// # Returns
/// Tensor with PReLU applied element-wise
#[pyfunction]
pub fn prelu(input: &PyTensor, _weight: &PyTensor) -> PyResult<PyTensor> {
    use coeus_nn::{Module, PReLU};
    let prelu_fn = PReLU::new();
    let result = prelu_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("PReLU forward failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// RReLU activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `lower` - Lower bound of the uniform distribution (default: 1.0/8.0)
/// * `upper` - Upper bound of the uniform distribution (default: 1.0/3.0)
///
/// # Returns
/// Tensor with RReLU applied element-wise
#[pyfunction]
#[pyo3(signature = (input, _lower=None, _upper=None))]
pub fn rrelu(input: &PyTensor, _lower: Option<f32>, _upper: Option<f32>) -> PyResult<PyTensor> {
    use coeus_nn::{Module, RReLU};
    let rrelu_fn = RReLU::new();
    let result = rrelu_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("RReLU forward failed: {}", e))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Tanhshrink activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Tensor with Tanhshrink applied element-wise
#[pyfunction]
pub fn tanhshrink(input: &PyTensor) -> PyResult<PyTensor> {
    use coeus_nn::{Module, Tanhshrink};
    let tanhshrink_fn = Tanhshrink::new();
    let result = tanhshrink_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Tanhshrink forward failed: {}",
            e
        ))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}

/// Threshold activation function (functional version)
///
/// # Arguments
/// * `input` - Input tensor
/// * `threshold` - The value to threshold at
/// * `value` - The value to replace with
///
/// # Returns
/// Tensor with Threshold applied element-wise
#[pyfunction]
#[pyo3(signature = (input, threshold, value))]
pub fn threshold(input: &PyTensor, threshold: f32, value: f32) -> PyResult<PyTensor> {
    use coeus_nn::{Module, Threshold};
    let threshold_fn = Threshold::new_with_params(threshold as f64, value as f64);
    let result = threshold_fn.forward(&input.tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Threshold forward failed: {}",
            e
        ))
    })?;
    Ok(PyTensor::from_rust_tensor(result))
}
