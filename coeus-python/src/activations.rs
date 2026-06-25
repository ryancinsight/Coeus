use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Element-wise ReLU activation.
#[pyfunction]
pub fn relu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::relu(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise Sigmoid activation.
#[pyfunction]
pub fn sigmoid(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sigmoid(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise Tanh activation.
#[pyfunction]
pub fn tanh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::tanh(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise GELU activation.
#[pyfunction]
pub fn gelu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::activation::gelu(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise SiLU activation.
#[pyfunction]
pub fn silu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::silu(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise Mish activation.
#[pyfunction]
pub fn mish(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::activation::mish(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise ELU activation.
#[pyfunction]
pub fn elu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::elu(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise Softplus activation.
#[pyfunction]
pub fn softplus(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softplus(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise GELU tanh approximation activation.
#[pyfunction]
pub fn gelu_tanh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::gelu_tanh(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise LeakyReLU activation.
#[pyfunction]
#[pyo3(signature = (input, negative_slope = 0.01))]
pub fn leaky_relu(input: &PyTensor, negative_slope: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::leaky_relu(&input.inner, negative_slope));
    PyTensor::from_var(inner)
}

/// Gated linear unit along a dimension.
#[pyfunction]
#[pyo3(signature = (input, dim = -1))]
pub fn glu(input: &PyTensor, dim: isize, py: Python<'_>) -> PyResult<PyTensor> {
    let shape = input.inner.tensor.shape();
    let ndim = shape.len();
    let dim = if dim < 0 {
        let normalized = ndim as isize + dim;
        if normalized < 0 {
            return Err(PyValueError::new_err(format!(
                "glu: dim {dim} out of range for rank {ndim}"
            )));
        }
        normalized as usize
    } else {
        dim as usize
    };
    if dim >= ndim {
        return Err(PyValueError::new_err(format!(
            "glu: dim {dim} out of range for rank {ndim}"
        )));
    }
    let axis = shape[dim];
    if !axis.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "glu: dim {dim} must have even size, got {axis}"
        )));
    }
    let half = axis / 2;
    let mut first_ranges: Vec<(usize, usize)> = shape.iter().map(|&extent| (0, extent)).collect();
    let mut second_ranges = first_ranges.clone();
    first_ranges[dim] = (0, half);
    second_ranges[dim] = (half, axis);
    let inner = py.allow_threads(|| {
        let first = coeus_autograd::slice(&input.inner, &first_ranges);
        let second = coeus_autograd::slice(&input.inner, &second_ranges);
        let gate = coeus_autograd::sigmoid(&second);
        coeus_autograd::mul(&first, &gate)
    });
    Ok(PyTensor::from_var(inner))
}
