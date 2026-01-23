use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

/// 1D convolution
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None))]
pub fn conv1d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<usize>,
    padding: Option<usize>,
) -> PyResult<PyTensor> {
    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => {
            let w = if let TensorWrapper::CpuDenseF32(inner) = &weight.inner {
                inner
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Weight must be F32",
                ));
            };
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
                coeus_nn::functional_api::conv1d(
                    i,
                    w,
                    b,
                    stride.unwrap_or(1),
                    padding.unwrap_or(0),
                )
                .map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "conv1d only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}

/// 2D convolution
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1))]
pub fn conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
    groups: usize,
) -> PyResult<PyTensor> {
    if dilation.unwrap_or((1, 1)) != (1, 1) {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv2d(dilation!=1) is not implemented",
        ));
    }
    if groups != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv2d(groups!=1) is not implemented",
        ));
    }

    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => {
            let w = if let TensorWrapper::CpuDenseF32(inner) = &weight.inner {
                inner
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Weight must be F32",
                ));
            };
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
                coeus_nn::functional_api::conv2d(i, w, b, stride, padding).map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "conv2d only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}

/// 2D transposed convolution (deconvolution)
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None))]
pub fn conv_transpose2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    output_padding: Option<(usize, usize)>,
    groups: usize,
    dilation: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    if dilation.unwrap_or((1, 1)) != (1, 1) {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv_transpose2d(dilation!=1) is not implemented",
        ));
    }
    if groups != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv_transpose2d(groups!=1) is not implemented",
        ));
    }

    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => {
            let w = if let TensorWrapper::CpuDenseF32(inner) = &weight.inner {
                inner
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Weight must be F32",
                ));
            };
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
                coeus_nn::functional_api::conv_transpose_2d(
                    i,
                    w,
                    b,
                    stride,
                    padding,
                    output_padding,
                )
                .map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "conv_transpose2d only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}

/// 3D convolution
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None))]
pub fn conv3d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize, usize)>,
    padding: Option<(usize, usize, usize)>,
) -> PyResult<PyTensor> {
    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(i) => {
            let w = if let TensorWrapper::CpuDenseF32(inner) = &weight.inner {
                inner
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Weight must be F32",
                ));
            };
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
                coeus_nn::functional_api::conv3d(
                    i,
                    w,
                    b,
                    stride.unwrap_or((1, 1, 1)),
                    padding.unwrap_or((0, 0, 0)),
                )
                .map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "conv3d only implemented for F32",
            ))
        }
    };
    Ok(PyTensor { inner: result })
}
