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
#[pyo3(signature = (input, chunks, dim = 0))]
pub fn chunk(
    input: &PyTensor,
    chunks: usize,
    dim: usize,
    py: Python<'_>,
) -> PyResult<Vec<PyTensor>> {
    let ndim = input.inner.tensor.ndim();
    if chunks == 0 {
        return Err(PyValueError::new_err(
            "chunk: chunks must be greater than zero",
        ));
    }
    if dim >= ndim {
        return Err(PyValueError::new_err(format!(
            "chunk: dim {dim} out of range for rank {ndim}"
        )));
    }
    let dim_size = input.inner.tensor.shape()[dim];
    if dim_size == 0 {
        return Ok(Vec::new());
    }
    let chunk_size = dim_size.div_ceil(chunks);
    let inner_chunks = py.allow_threads(|| coeus_autograd::split(&input.inner, chunk_size, dim));
    Ok(inner_chunks.into_iter().map(PyTensor::from_var).collect())
}

#[pyfunction]
pub fn one_hot(input: &PyTensor, num_classes: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if input.inner.tensor.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "one_hot: indices must be 1-D, got {}-D",
            input.inner.tensor.ndim()
        )));
    }
    let indices = input.inner.tensor.to_contiguous();
    for &value in indices.as_slice() {
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
            return Err(PyValueError::new_err(format!(
                "one_hot: index value {value} is not a non-negative integer"
            )));
        }
        let idx = value as usize;
        if idx >= num_classes {
            return Err(PyValueError::new_err(format!(
                "one_hot: index {idx} out of range for num_classes={num_classes}"
            )));
        }
    }
    let backend = MoiraiBackend::new();
    let tensor = py.allow_threads(|| {
        coeus_ops::one_hot::<f64, MoiraiBackend>(&input.inner.tensor, num_classes, &backend)
    });
    Ok(PyTensor::from_var(coeus_autograd::Var::new(tensor, false)))
}

#[pyfunction]
pub fn masked_select(input: &PyTensor, mask: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    if input.inner.tensor.shape() != mask.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "masked_select: input shape {:?} must match mask shape {:?}",
            input.inner.tensor.shape(),
            mask.inner.tensor.shape()
        )));
    }
    let backend = MoiraiBackend::new();
    let tensor = py.allow_threads(|| {
        coeus_ops::masked_select::<f64, MoiraiBackend>(
            &input.inner.tensor,
            &mask.inner.tensor,
            &backend,
        )
    });
    Ok(PyTensor::from_var(coeus_autograd::Var::new(tensor, false)))
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

/// Scatter-assign `values` into `input` at row indices given by `indices`.
///
/// Equivalent to `torch.index_put(input, (indices,), values, accumulate)`.
#[pyfunction]
#[pyo3(signature = (input, indices, values, accumulate = false))]
pub fn index_put(
    input: &PyTensor,
    indices: &PyTensor,
    values: &PyTensor,
    accumulate: bool,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if indices.inner.tensor.ndim() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "index_put: indices must be 1-D, got {}-D",
            indices.inner.tensor.ndim()
        )));
    }
    let x = input.inner.clone();
    let idx = indices.inner.clone();
    let vals = values.inner.clone();
    let inner = py.allow_threads(move || {
        let backend = MoiraiBackend::new();
        let t = coeus_ops::index_put(&x.tensor, &idx.tensor, &vals.tensor, accumulate, &backend);
        coeus_autograd::Var::new(t, false)
    });
    Ok(PyTensor::from_var(inner))
}
