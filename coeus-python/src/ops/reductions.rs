use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn sum_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sum_axis(&input.inner, axis));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn mean_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::mean_axis(&input.inner, axis));
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
pub fn prod(input: &PyTensor, py: Python<'_>) -> f64 {
    py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        coeus_ops::prod::<f64, MoiraiBackend>(&input.inner.tensor, &backend)
    })
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
