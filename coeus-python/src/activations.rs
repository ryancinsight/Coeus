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

/// Element-wise SELU activation.
#[pyfunction]
pub fn selu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::selu(&input.inner));
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
    let normalized = if dim < 0 { ndim as isize + dim } else { dim };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(PyValueError::new_err(format!(
            "glu: dim {dim} out of range for rank {ndim}"
        )));
    }
    let dim = normalized as usize;
    if !shape[dim].is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "glu: dim {dim} must have even size, got {}",
            shape[dim]
        )));
    }
    // GLU algorithm lives in coeus-nn; the binding only normalizes the Python
    // negative-dim convention and maps invalid arguments to a Python exception.
    let inner = py.allow_threads(|| coeus_nn::glu(&input.inner, dim));
    Ok(PyTensor::from_var(inner))
}

/// Masked softmax over `dim`: softmax across positions where `mask != 0`; masked
/// positions (and any fully-masked row) are zero. Differentiable in `input`.
///
/// `mask` shape must match `input` shape.
#[pyfunction]
#[pyo3(signature = (input, mask, dim = -1))]
pub fn masked_softmax(
    input: &PyTensor,
    mask: &PyTensor,
    dim: i64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let ndim = input.inner.tensor.ndim() as i64;
    let axis = if dim < 0 { ndim + dim } else { dim };
    if axis < 0 || axis >= ndim {
        return Err(PyValueError::new_err(format!(
            "masked_softmax: dim {dim} out of range for rank {ndim}"
        )));
    }
    if input.inner.tensor.shape() != mask.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "masked_softmax: input shape {:?} must match mask shape {:?}",
            input.inner.tensor.shape(),
            mask.inner.tensor.shape()
        )));
    }
    let inner = py.allow_threads(|| {
        coeus_autograd::masked_softmax(&input.inner, &mask.inner.tensor, axis as isize)
    });
    Ok(PyTensor::from_var(inner))
}

/// Causal (lower-triangular) softmax over `dim`: future positions are masked before
/// softmax. Differentiable in `input`.
#[pyfunction]
#[pyo3(signature = (input, dim = -1))]
pub fn causal_softmax(input: &PyTensor, dim: i64, py: Python<'_>) -> PyResult<PyTensor> {
    let ndim = input.inner.tensor.ndim() as i64;
    let axis = if dim < 0 { ndim + dim } else { dim };
    if axis < 0 || axis >= ndim {
        return Err(PyValueError::new_err(format!(
            "causal_softmax: dim {dim} out of range for rank {ndim}"
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::causal_softmax(&input.inner, axis as isize));
    Ok(PyTensor::from_var(inner))
}

// ── G-037 extended activation family ──

/// Element-wise Hardtanh activation (default range `[-1, 1]`).
#[pyfunction]
#[pyo3(signature = (input, min_val = -1.0, max_val = 1.0))]
pub fn hardtanh(input: &PyTensor, min_val: f64, max_val: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::hardtanh(&input.inner, min_val, max_val));
    PyTensor::from_var(inner)
}

/// Element-wise Hardsigmoid activation.
#[pyfunction]
pub fn hardsigmoid(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::hardsigmoid(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise Hardswish activation.
#[pyfunction]
pub fn hardswish(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::hardswish(&input.inner));
    PyTensor::from_var(inner)
}

/// Element-wise Hardshrink activation.
#[pyfunction]
#[pyo3(signature = (input, lambd = 0.5))]
pub fn hardshrink(input: &PyTensor, lambd: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::hardshrink(&input.inner, lambd));
    PyTensor::from_var(inner)
}

/// Element-wise Softshrink activation.
#[pyfunction]
#[pyo3(signature = (input, lambd = 0.5))]
pub fn softshrink(input: &PyTensor, lambd: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softshrink(&input.inner, lambd));
    PyTensor::from_var(inner)
}

/// Element-wise Softsign activation.
#[pyfunction]
pub fn softsign(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softsign(&input.inner));
    PyTensor::from_var(inner)
}

/// Threshold activation: `x if x > threshold else value`.
#[pyfunction]
#[pyo3(signature = (input, threshold, value))]
pub fn threshold(input: &PyTensor, threshold: f64, value: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::threshold(&input.inner, threshold, value));
    PyTensor::from_var(inner)
}

/// Celu activation: `max(0,x) + min(0, α·(exp(x/α)-1))`.
#[pyfunction]
#[pyo3(signature = (input, alpha = 1.0))]
pub fn celu(input: &PyTensor, alpha: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::celu(&input.inner, alpha));
    PyTensor::from_var(inner)
}

/// PReLU activation (single scalar α, non-learnable through this functional
/// entry point).
///
/// `coeus_autograd::prelu` now takes a learnable weight `Var` (matching
/// PyTorch/Burn `nn.PReLU`'s per-channel learnable slope); this binding
/// constructs a fixed (`requires_grad = false`) scalar weight internally so
/// the existing `prelu(input, alpha: float)` Python surface keeps working
/// unchanged. A `PReLU` module class exposing the learnable weight as a
/// registered parameter is deferred to the Python binding pass (blocked on
/// the coeus-python wheel build).
#[pyfunction]
#[pyo3(signature = (input, alpha = 0.25))]
pub fn prelu(input: &PyTensor, alpha: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| {
        let backend = coeus_core::MoiraiBackend::new();
        let weight = coeus_autograd::Var::new(
            coeus_tensor::Tensor::full_on([1], alpha, &backend),
            false,
        );
        coeus_autograd::prelu(&input.inner, &weight)
    });
    PyTensor::from_var(inner)
}

/// LogSigmoid activation: `log(sigmoid(x))`, via the stable `-softplus(-x)`.
#[pyfunction]
pub fn log_sigmoid(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::log_sigmoid(&input.inner));
    PyTensor::from_var(inner)
}

/// Tanhshrink activation: `x - tanh(x)`.
#[pyfunction]
pub fn tanhshrink(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::tanhshrink(&input.inner));
    PyTensor::from_var(inner)
}
