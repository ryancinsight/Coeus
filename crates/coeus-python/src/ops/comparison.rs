use crate::tensor::PyTensor;
use pyo3::prelude::*;

#[pyfunction]
pub fn eq(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::eq(&a.inner, &b.inner));
    PyTensor { inner }
}

#[pyfunction]
pub fn ne(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::ne(&a.inner, &b.inner));
    PyTensor { inner }
}

#[pyfunction]
pub fn lt(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::lt(&a.inner, &b.inner));
    PyTensor { inner }
}

#[pyfunction]
pub fn gt(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::gt(&a.inner, &b.inner));
    PyTensor { inner }
}

#[pyfunction]
pub fn le(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::le(&a.inner, &b.inner));
    PyTensor { inner }
}

#[pyfunction]
pub fn ge(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::ge(&a.inner, &b.inner));
    PyTensor { inner }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_ref_mut, reason = "ratchet COEUS-LINT-1")]
pub fn where_fn(
    cond: &PyTensor,
    on_true: &PyTensor,
    on_false: &PyTensor,
    py: Python<'_>,
) -> PyTensor {
    let inner = py
        .allow_threads(|| coeus_autograd::where_cond(&cond.inner, &on_true.inner, &on_false.inner));
    PyTensor::from_var(inner)
}
