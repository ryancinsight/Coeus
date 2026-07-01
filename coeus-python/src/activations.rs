use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
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

/// Masked softmax: replaces `mask==0` positions with `-inf` before softmax.
///
/// `mask` shape must match `input` shape. Equivalent to
/// `torch.softmax(input.masked_fill(~mask, float('-inf')), dim=dim)`.
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
    let x = input.inner.tensor.clone();
    let m = mask.inner.tensor.clone();
    let ax = axis as usize;
    let inner = py.allow_threads(move || {
        let backend = MoiraiBackend::new();
        coeus_ops::masked_softmax(&x, &m, ax, &backend)
    });
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(inner, false),
    })
}

/// Causal (lower-triangular) softmax — masks future positions before softmax.
///
/// Equivalent to `torch.softmax(input.masked_fill(upper_tri_mask, -inf), dim)`.
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
    let x = input.inner.tensor.clone();
    let ax = axis as usize;
    let inner = py.allow_threads(move || {
        let backend = MoiraiBackend::new();
        coeus_ops::causal_softmax(&x, ax, &backend)
    });
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(inner, false),
    })
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

/// PReLU activation (single scalar α).
#[pyfunction]
#[pyo3(signature = (input, alpha = 0.25))]
pub fn prelu(input: &PyTensor, alpha: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::prelu(&input.inner, alpha));
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
