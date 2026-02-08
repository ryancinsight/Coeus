use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use dtype::float::Float32;
use pyo3::prelude::*;

#[pyfunction]
pub fn relu(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::relu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::relu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::relu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF32(t) => {
            // relu currently returns sparse, convert to dense
            let sparse_res = coeus_nn::functional_api::relu(t).map_err(to_py_err)?;
            let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF64(t) => {
            let sparse_res = coeus_nn::functional_api::relu(t).map_err(to_py_err)?;
            let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "relu not implemented for integer tensors",
        )),
    }
}

#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF32(t) => {
            // sigmoid returns dense (sigmoid(0)!=0)
            let sparse_res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
            let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF64(t) => {
            let sparse_res = coeus_nn::functional_api::sigmoid(t).map_err(to_py_err)?;
            let res = sparse_res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "sigmoid not implemented for integer tensors",
        )),
    }
}

#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::tanh(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::tanh(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::tanh(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF32(t) => {
            let res = coeus_nn::functional_api::tanh(t).map_err(to_py_err)?;
            let dense = res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(dense),
            })
        }
        TensorWrapper::CpuSparseF64(t) => {
            let res = coeus_nn::functional_api::tanh(t).map_err(to_py_err)?;
            let dense = res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(dense),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "tanh not implemented for integer tensors",
        )),
    }
}

#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::gelu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::gelu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::gelu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF32(t) => {
            let res = coeus_nn::functional_api::gelu(t).map_err(to_py_err)?;
            let dense = res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(dense),
            })
        }
        TensorWrapper::CpuSparseF64(t) => {
            let res = coeus_nn::functional_api::gelu(t).map_err(to_py_err)?;
            let dense = res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(dense),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "gelu not implemented for this tensor type",
        )),
    }
}

#[pyfunction]
pub fn silu(input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::silu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::silu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::silu(t).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        TensorWrapper::CpuSparseF32(t) => {
            let res = coeus_nn::functional_api::silu(t).map_err(to_py_err)?;
            let dense = res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(dense),
            })
        }
        TensorWrapper::CpuSparseF64(t) => {
            let res = coeus_nn::functional_api::silu(t).map_err(to_py_err)?;
            let dense = res.to_dense_generic().map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(dense),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "silu not implemented for this tensor type",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, negative_slope=0.01))]
pub fn leaky_relu(input: &PyTensor, negative_slope: f64) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(inner) => {
            let res =
                coeus_nn::functional_api::leaky_relu(inner, Float32::new(negative_slope as f32))
                    .map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "leaky_relu only implemented for F32",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, alpha=1.0))]
pub fn elu(input: &PyTensor, alpha: f64) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(inner) => {
            let res = coeus_nn::functional_api::elu(inner, Float32::new(alpha as f32))
                .map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "elu only implemented for F32",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, dtype=None))]
pub fn softmax(
    input: &PyTensor,
    dim: Option<isize>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    if dtype.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "softmax(dtype=...) is not implemented",
        ));
    }

    let dim = dim.unwrap_or(-1);
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::softmax_dim(t, dim).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::softmax_dim(t, dim).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::softmax_dim(t, dim).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "softmax only implemented for dense float tensors",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, dtype=None))]
pub fn log_softmax(
    input: &PyTensor,
    dim: Option<isize>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    if dtype.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "log_softmax(dtype=...) is not implemented",
        ));
    }

    let dim = dim.unwrap_or(-1);
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::log_softmax(t, dim).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::log_softmax(t, dim).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(res),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::log_softmax(t, dim).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "log_softmax only implemented for dense float tensors",
        )),
    }
}
