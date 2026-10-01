use crate::{error::map_backend_error, tensor::PyTensor};
use coeus_autograd::Var;
use coeus_core::BackendError;
use pyo3::prelude::*;

#[inline]
fn comparison_dispatch<F>(a: &PyTensor, b: &PyTensor, py: Python<'_>, op: F) -> PyResult<PyTensor>
where
    F: FnOnce(&Var<f64>, &Var<f64>) -> Result<Var<f64>, BackendError> + Send,
{
    let inner = py.allow_threads(|| op(&a.inner, &b.inner));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

macro_rules! comparison_fn {
    ($name:ident, $op:path) => {
        #[pyfunction]
        pub fn $name(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
            comparison_dispatch(a, b, py, $op)
        }
    };
}

comparison_fn!(eq, coeus_autograd::eq);
comparison_fn!(ne, coeus_autograd::ne);
comparison_fn!(lt, coeus_autograd::lt);
comparison_fn!(gt, coeus_autograd::gt);
comparison_fn!(le, coeus_autograd::le);
comparison_fn!(ge, coeus_autograd::ge);

#[pyfunction]
pub fn where_fn(
    cond: &PyTensor,
    on_true: &PyTensor,
    on_false: &PyTensor,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let inner = py
        .allow_threads(|| coeus_autograd::where_cond(&cond.inner, &on_true.inner, &on_false.inner));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}
