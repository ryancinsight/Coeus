use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (input, axis, keepdim = false))]
pub fn sum_axis(input: &PyTensor, axis: usize, keepdim: bool, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| {
        let out = coeus_autograd::sum_axis(&input.inner, axis);
        if keepdim {
            out
        } else {
            // sum_axis always keeps the dim; squeeze it if keepdim=False.
            coeus_autograd::squeeze(&out, Some(axis))
        }
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (input, axis, keepdim = false))]
pub fn mean_axis(input: &PyTensor, axis: usize, keepdim: bool, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| {
        let out = coeus_autograd::mean_axis(&input.inner, axis);
        if keepdim {
            out
        } else {
            coeus_autograd::squeeze(&out, Some(axis))
        }
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn log_softmax(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log_softmax(&input.inner, axis));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn cumsum(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::cumsum(&input.inner, dim));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn max_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::max_axis(&input.inner, axis));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn min_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::min_axis(&input.inner, axis));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn log_sum_exp(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log_sum_exp(&input.inner, axis));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn sum(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sum(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn mean(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::mean(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn argmax(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "argmax: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let backend = MoiraiBackend::new();
    let idx_i64 =
        py.allow_threads(|| coeus_ops::argmax::<f64, MoiraiBackend>(&input.inner.tensor, dim));
    let data: Vec<f64> = idx_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let t = Tensor::<f64, MoiraiBackend>::from_slice(idx_i64.shape().to_vec(), &data);
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    })
}

#[pyfunction]
pub fn argmin(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "argmin: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let backend = MoiraiBackend::new();
    let idx_i64 =
        py.allow_threads(|| coeus_ops::argmin::<f64, MoiraiBackend>(&input.inner.tensor, dim));
    let data: Vec<f64> = idx_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let t = Tensor::<f64, MoiraiBackend>::from_slice(idx_i64.shape().to_vec(), &data);
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    })
}

#[pyfunction]
pub fn amax(input: &PyTensor, py: Python<'_>) -> PyResult<f64> {
    if input.inner.tensor.numel() == 0 {
        return Err(PyValueError::new_err("amax: empty tensor has no maximum"));
    }
    let v = py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        coeus_ops::amax::<f64, MoiraiBackend>(&input.inner.tensor, &backend)
    });
    Ok(v)
}

#[pyfunction]
pub fn amin(input: &PyTensor, py: Python<'_>) -> PyResult<f64> {
    if input.inner.tensor.numel() == 0 {
        return Err(PyValueError::new_err("amin: empty tensor has no minimum"));
    }
    let v = py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        coeus_ops::amin::<f64, MoiraiBackend>(&input.inner.tensor, &backend)
    });
    Ok(v)
}

#[pyfunction]
pub fn prod(input: &PyTensor, py: Python<'_>) -> PyTensor {
    // Tracked composition (cumprod + slice) so gradients flow
    // (d prod/dx_i = prod_{j != i} x_j), matching torch.prod; returns [1].
    let inner = py.allow_threads(|| coeus_autograd::prod(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn cumprod(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "cumprod: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::cumprod(&input.inner, dim));
    Ok(PyTensor::from_var(inner))
}

/// Lp-normalize `input` along `dim`.
///
/// `output[i] = input[i] / max(norm_p(input, p, dim), eps)`
///
/// Equivalent to `torch.nn.functional.normalize(input, p=2, dim=1)`.
#[pyfunction]
#[pyo3(signature = (input, p = 2.0, dim = 1, eps = 1e-12))]
pub fn normalize(
    input: &PyTensor,
    p: f64,
    dim: usize,
    eps: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "normalize: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    if p <= 0.0 || !p.is_finite() {
        return Err(PyValueError::new_err(format!(
            "normalize: p must be positive finite, got {p}"
        )));
    }
    let x = input.inner.clone();
    let inner = py.allow_threads(move || {
        let norms = coeus_autograd::norm_p_axis(&x, p, dim);
        let shifted_norms = coeus_autograd::scalar_sub(&norms, eps);
        let norms_clamped = coeus_autograd::scalar_add(&coeus_autograd::relu(&shifted_norms), eps);
        coeus_autograd::div(&x, &norms_clamped)
    });
    Ok(PyTensor::from_var(inner))
}

/// Element-wise closeness test.
///
/// Returns a float tensor (1.0 = close, 0.0 = not close).
/// `|a - b| <= atol + rtol * |b|`
#[pyfunction]
#[pyo3(signature = (a, b, rtol = 1e-5, atol = 1e-8))]
pub fn isclose(
    a: &PyTensor,
    b: &PyTensor,
    rtol: f64,
    atol: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if a.inner.tensor.shape() != b.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "isclose: shape mismatch {:?} vs {:?}",
            a.inner.tensor.shape(),
            b.inner.tensor.shape()
        )));
    }
    let av = a.inner.tensor.to_contiguous();
    let bv = b.inner.tensor.to_contiguous();
    let as_ = av.as_slice();
    let bs = bv.as_slice();
    let data: Vec<f64> = as_
        .iter()
        .zip(bs.iter())
        .map(|(&ai, &bi)| {
            let close = (ai - bi).abs() <= atol + rtol * bi.abs();
            if close {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let _ = py;
    let t = Tensor::<f64, MoiraiBackend>::from_slice(a.inner.tensor.shape().to_vec(), &data);
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    })
}

/// Returns `true` if all elements are element-wise close.
///
/// Equivalent to `torch.allclose(a, b, rtol, atol)`.
#[pyfunction]
#[pyo3(signature = (a, b, rtol = 1e-5, atol = 1e-8))]
pub fn allclose(
    a: &PyTensor,
    b: &PyTensor,
    rtol: f64,
    atol: f64,
    py: Python<'_>,
) -> PyResult<bool> {
    let close = isclose(a, b, rtol, atol, py)?;
    let data = close.inner.tensor.to_contiguous();
    Ok(data.as_slice().iter().all(|&v| v != 0.0))
}

/// Replace NaN and Inf values with finite numbers.
///
/// Equivalent to `torch.nan_to_num(input, nan, posinf, neginf)`.
#[pyfunction]
#[pyo3(signature = (input, nan = 0.0, posinf = None, neginf = None))]
pub fn nan_to_num(
    input: &PyTensor,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
    py: Python<'_>,
) -> PyTensor {
    let pos_val = posinf.unwrap_or(f64::MAX);
    let neg_val = neginf.unwrap_or(f64::MIN);
    let x = input.inner.clone();
    let data = py.allow_threads(move || {
        let cont = x.tensor.to_contiguous();
        cont.as_slice()
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    nan
                } else if v.is_infinite() && v > 0.0 {
                    pos_val
                } else if v.is_infinite() && v < 0.0 {
                    neg_val
                } else {
                    v
                }
            })
            .collect::<Vec<f64>>()
    });
    let shape = input.inner.tensor.shape().to_vec();
    let t = Tensor::<f64, MoiraiBackend>::from_slice(shape, &data);
    PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    }
}
