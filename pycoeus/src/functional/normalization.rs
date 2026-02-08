use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;


/// Layer normalization function
#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, bias=None, eps=1e-5))]
pub fn layer_norm(
    input: &PyTensor,
    normalized_shape: Vec<usize>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: Option<f64>,
) -> PyResult<PyTensor> {
    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => {
            let w = weight
                .map(|t| {
                    if let TensorWrapper::CpuDenseF32(inner) = &t.inner {
                        Ok(inner)
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Weight must be F32",
                        ))
                    }
                })
                .transpose()?;
            let b = bias
                .map(|t| {
                    if let TensorWrapper::CpuDenseF32(inner) = &t.inner {
                        Ok(inner)
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Bias must be F32",
                        ))
                    }
                })
                .transpose()?;
            TensorWrapper::CpuDenseF32(
                coeus_nn::functional_api::layer_norm(
                    i,
                    &normalized_shape,
                    w,
                    b,
                    eps.unwrap_or(1e-5),
                )
                .map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "layer_norm only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, running_mean=None, running_var=None, weight=None, bias=None, training=false, momentum=0.1, eps=1e-5))]
pub fn batch_norm(
    input: &PyTensor,
    running_mean: Option<&PyTensor>,
    running_var: Option<&PyTensor>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> PyResult<PyTensor> {
    if running_mean.is_some() || running_var.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "batch_norm with running statistics is not implemented",
        ));
    }

    if !training {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "batch_norm(training=False) is not implemented (requires running statistics)",
        ));
    }

    if (momentum - 0.1).abs() > 1e-12 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "batch_norm(momentum!=0.1) is not implemented",
        ));
    }

    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => {
            let w = weight
                .map(|t| {
                    if let TensorWrapper::CpuDenseF32(inner) = &t.inner {
                        Ok(inner)
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Weight must be F32",
                        ))
                    }
                })
                .transpose()?;
            let b = bias
                .map(|t| {
                    if let TensorWrapper::CpuDenseF32(inner) = &t.inner {
                        Ok(inner)
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Bias must be F32",
                        ))
                    }
                })
                .transpose()?;
            TensorWrapper::CpuDenseF32(
                coeus_nn::functional_api::batch_norm(i, w, b, eps).map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "batch_norm only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}
