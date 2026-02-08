use crate::tensor::class::PyTensor;
use pyo3::prelude::*;

#[pyfunction]
pub fn matmul(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.matmul(other)
}

#[pyfunction]
pub fn bmm(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.bmm(other)
}

#[pyfunction]
pub fn addmm(
    input: &PyTensor,
    mat1: &PyTensor,
    mat2: &PyTensor,
    beta: f32,
    alpha: f32,
) -> PyResult<PyTensor> {
    input.addmm(mat1, mat2, beta as f64, alpha as f64)
}

#[pyfunction]
pub fn mv(input: &PyTensor, vec: &PyTensor) -> PyResult<PyTensor> {
    input.mv(vec)
}

#[pyfunction]
pub fn addr(input: &PyTensor, vec1: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
    input.addr(vec1, vec2, 1.0, 1.0)
}

#[pyfunction]
pub fn outer(input: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
    input.outer(vec2)
}

#[pyfunction]
pub fn reshape(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.reshape(shape)
}

#[pyfunction]
pub fn view(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.view(shape)
}

#[pyfunction]
pub fn flatten(input: &PyTensor, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
    input.flatten(start_dim, end_dim)
}

#[pyfunction]
pub fn squeeze(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    input.squeeze(dim)
}

#[pyfunction]
pub fn unsqueeze(input: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    input.unsqueeze(dim)
}

#[pyfunction]
pub fn transpose(input: &PyTensor, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
    input.transpose(dim0, dim1)
}

#[pyfunction]
pub fn permute(input: &PyTensor, dims: Vec<usize>) -> PyResult<PyTensor> {
    input.permute_internal(dims)
}

#[pyfunction]
pub fn dropout(_input: &PyTensor, _p: f32, _training: bool) -> PyResult<PyTensor> {
    Err(crate::tensor::class::to_py_err(
        "Dropout functional not yet implemented",
    ))
}

#[pyfunction]
pub fn argmax(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.argmax(dim, keepdim)
}

#[pyfunction]
pub fn argmin(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    input.argmin(dim, keepdim)
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(bmm, m)?)?;
    m.add_function(wrap_pyfunction!(addmm, m)?)?;
    m.add_function(wrap_pyfunction!(mv, m)?)?;
    m.add_function(wrap_pyfunction!(addr, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(view, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(unsqueeze, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(permute, m)?)?;
    m.add_function(wrap_pyfunction!(dropout, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(gather, m)?)?;
    m.add_function(wrap_pyfunction!(index_select, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(index_add, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn gather(input: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    crate::tensor::ops::indexing::gather(input, dim, index)
}

#[pyfunction]
pub fn index_select(input: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    crate::tensor::ops::indexing::index_select(input, dim, index)
}

#[pyfunction]
pub fn nonzero(input: &PyTensor) -> PyResult<PyTensor> {
    crate::tensor::ops::indexing::nonzero(input)
}

#[pyfunction]
#[pyo3(signature = (input, dim, index, source, alpha=None))]
pub fn index_add(input: &PyTensor, dim: usize, index: &PyTensor, source: &PyTensor, alpha: Option<f64>) -> PyResult<PyTensor> {
    crate::tensor::ops::indexing::index_add(input, dim, index, source, alpha.unwrap_or(1.0))
}

#[pyfunction]
#[pyo3(signature = (x1, x2, p=2.0))]
pub fn pairwise_distance(x1: &PyTensor, x2: &PyTensor, p: f64) -> PyResult<PyTensor> {
    crate::functional::distance::pairwise_distance(x1, x2, p, 1e-6, false)
}

#[pyfunction]
#[pyo3(signature = (x1, x2, dim=1, eps=1e-8))]
pub fn cosine_similarity(x1: &PyTensor, x2: &PyTensor, dim: usize, eps: f64) -> PyResult<PyTensor> {
    crate::functional::distance::cosine_similarity(x1, x2, dim, eps)
}
