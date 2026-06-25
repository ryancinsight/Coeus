use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn gather(input: &PyTensor, dim: usize, index: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::gather(&input.inner, dim, &index.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn scatter_add(
    input: &PyTensor,
    dim: usize,
    index: &PyTensor,
    src: &PyTensor,
    py: Python<'_>,
) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = py.allow_threads(|| {
        coeus_ops::scatter_add(
            &input.inner.tensor,
            dim,
            &index.inner.tensor,
            &src.inner.tensor,
            &backend,
        )
    });
    PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    }
}

#[pyfunction]
pub fn repeat_interleave(input: &PyTensor, repeats: usize, dim: usize, py: Python<'_>) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = py.allow_threads(|| {
        coeus_ops::repeat_interleave(&input.inner.tensor, repeats, dim, &backend)
    });
    PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    }
}

#[pyfunction]
pub fn index_select(
    input: &PyTensor,
    dim: usize,
    index: &PyTensor,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if index.inner.tensor.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "index_select: index must be 1-D, got {}-D",
            index.inner.tensor.ndim()
        )));
    }
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "index_select: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::index_select(&input.inner, dim, &index.inner));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn nonzero(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = py
        .allow_threads(|| coeus_ops::nonzero::<f64, MoiraiBackend>(&input.inner.tensor, &backend));
    PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    }
}

#[pyfunction]
pub fn masked_fill(
    input: &PyTensor,
    mask: &PyTensor,
    value: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if input.inner.tensor.shape() != mask.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "masked_fill: input shape {:?} must match mask shape {:?}",
            input.inner.tensor.shape(),
            mask.inner.tensor.shape()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::masked_fill(&input.inner, &mask.inner, value));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn where_cond(
    cond: &PyTensor,
    on_true: &PyTensor,
    on_false: &PyTensor,
    py: Python<'_>,
) -> PyTensor {
    let inner = py
        .allow_threads(|| coeus_autograd::where_cond(&cond.inner, &on_true.inner, &on_false.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (input, k = 0))]
pub fn tril(input: &PyTensor, k: i64, py: Python<'_>) -> PyResult<PyTensor> {
    let ndim = input.inner.tensor.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err(format!(
            "tril: requires at least 2-D input, got {ndim}-D"
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::tril(&input.inner, k as isize));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
#[pyo3(signature = (input, k = 0))]
pub fn triu(input: &PyTensor, k: i64, py: Python<'_>) -> PyResult<PyTensor> {
    let ndim = input.inner.tensor.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err(format!(
            "triu: requires at least 2-D input, got {ndim}-D"
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::triu(&input.inner, k as isize));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn roll(
    input: &PyTensor,
    shifts: Vec<i64>,
    dims: Vec<usize>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if shifts.len() != dims.len() {
        return Err(PyValueError::new_err(format!(
            "roll: shifts ({}) and dims ({}) must have equal length",
            shifts.len(),
            dims.len()
        )));
    }
    let ndim = input.inner.tensor.ndim();
    for &d in &dims {
        if d >= ndim {
            return Err(PyValueError::new_err(format!(
                "roll: dim {d} out of range for {ndim}-D tensor"
            )));
        }
    }
    let shifts_isize: Vec<isize> = shifts.iter().map(|&s| s as isize).collect();
    let inner = py.allow_threads(move || coeus_autograd::roll(&input.inner, &shifts_isize, &dims));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn split(input: &PyTensor, chunk_size: usize, dim: usize, py: Python<'_>) -> Vec<PyTensor> {
    let inner_chunks = py.allow_threads(|| coeus_autograd::split(&input.inner, chunk_size, dim));
    inner_chunks.into_iter().map(PyTensor::from_var).collect()
}

#[pyfunction]
pub fn cat(inputs: Vec<pyo3::Py<PyTensor>>, dim: usize, py: Python<'_>) -> PyTensor {
    let rust_inputs: Vec<coeus_autograd::Var<f64>> = inputs
        .iter()
        .map(|t| t.bind(py).borrow().inner.clone())
        .collect();
    let inner = py.allow_threads(move || {
        let ref_inputs: Vec<&coeus_autograd::Var<f64>> = rust_inputs.iter().collect();
        coeus_autograd::cat(&ref_inputs, dim)
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn stack(inputs: Vec<pyo3::Py<PyTensor>>, dim: usize, py: Python<'_>) -> PyTensor {
    let rust_inputs: Vec<coeus_autograd::Var<f64>> = inputs
        .iter()
        .map(|t| t.bind(py).borrow().inner.clone())
        .collect();
    let inner = py.allow_threads(move || {
        let ref_inputs: Vec<&coeus_autograd::Var<f64>> = rust_inputs.iter().collect();
        coeus_autograd::stack(&ref_inputs, dim)
    });
    PyTensor::from_var(inner)
}
