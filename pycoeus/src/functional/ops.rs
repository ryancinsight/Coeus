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
    input.addr(vec1, vec2)
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
    input.permute(dims)
}

#[pyfunction]
pub fn dropout(_input: &PyTensor, _p: f32, _training: bool) -> PyResult<PyTensor> {
    Err(crate::tensor::class::to_py_err(
        "Dropout functional not yet implemented",
    ))
}

#[pyfunction]
pub fn argmax(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    crate::tensor::ops::reduction::argmax(input, dim, keepdim)
}

#[pyfunction]
pub fn argmin(input: &PyTensor, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
    crate::tensor::ops::reduction::argmin(input, dim, keepdim)
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
    Ok(())
}
