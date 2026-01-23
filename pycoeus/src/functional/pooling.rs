use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

/// Max pooling 2D
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None))]
pub fn max_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => TensorWrapper::CpuDenseF32(
            coeus_nn::functional_api::max_pool2d(i, kernel_size, stride, padding.unwrap_or((0, 0)))
                .map_err(to_py_err)?,
        ),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "max_pool2d only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}

/// Average pooling 2D
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None))]
pub fn avg_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => TensorWrapper::CpuDenseF32(
            coeus_nn::functional_api::avg_pool2d(i, kernel_size, stride, padding.unwrap_or((0, 0)))
                .map_err(to_py_err)?,
        ),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "avg_pool2d only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}
