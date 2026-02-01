use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

#[pyfunction]
pub fn eig(input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let (eigenvalues, eigenvectors) = tensor::ops::eig(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(eigenvalues),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(eigenvectors),
                },
            ))
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let (eigenvalues, eigenvectors) = tensor::ops::eig(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(eigenvalues),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(eigenvectors),
                },
            ))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "eig not implemented for this dtype",
        )),
    }
}

#[pyfunction]
pub fn eigh(input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let (eigenvalues, eigenvectors) = tensor::ops::eigh(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(eigenvalues),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(eigenvectors),
                },
            ))
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let (eigenvalues, eigenvectors) = tensor::ops::eigh(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(eigenvalues),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(eigenvectors),
                },
            ))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "eigh not implemented for this dtype",
        )),
    }
}

#[pyfunction]
pub fn matrix_exp(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let result = tensor::ops::matrix_exp(tensor).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let result = tensor::ops::matrix_exp(tensor).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "matrix_exp not implemented for this dtype",
        )),
    }
}

#[pyfunction]
pub fn matrix_power(input: &PyTensor, n: i64) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let result = tensor::ops::matrix_power(tensor, n).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let result = tensor::ops::matrix_power(tensor, n).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "matrix_power not implemented for this dtype",
        )),
    }
}
#[pyfunction]
pub fn cholesky(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let result = tensor::ops::cholesky(tensor).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let result = tensor::ops::cholesky(tensor).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cholesky not implemented for this dtype",
        )),
    }
}

#[pyfunction]
pub fn qr(input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let (q, r) = tensor::ops::qr(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(q),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(r),
                },
            ))
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let (q, r) = tensor::ops::qr(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(q),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(r),
                },
            ))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "qr not implemented for this dtype",
        )),
    }
}

#[pyfunction]
pub fn svd(input: &PyTensor) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(tensor) => {
            let (u, s, v) = tensor::ops::svd(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(u),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(s),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF32(v),
                },
            ))
        }
        TensorWrapper::CpuDenseF64(tensor) => {
            let (u, s, v) = tensor::ops::svd(tensor).map_err(to_py_err)?;
            Ok((
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(u),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(s),
                },
                PyTensor {
                    inner: TensorWrapper::CpuDenseF64(v),
                },
            ))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "svd not implemented for this dtype",
        )),
    }
}
