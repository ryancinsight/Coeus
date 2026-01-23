//! Python bindings for special functions.
//!
//! This module exposes special mathematical functions to Python via PyO3,
//! using TensorWrapper dispatch pattern for backend/dtype flexibility.

use crate::tensor::{PyTensor, TensorWrapper};
use coeus_special::bessel::Bessel;
use coeus_special::error_functions::Erf;
use coeus_special::gamma::Gamma;
use coeus_special::trig::Misc;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(erfinv, m)?)?;
    m.add_function(wrap_pyfunction!(ndtr, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(lgamma, m)?)?;
    m.add_function(wrap_pyfunction!(digamma, m)?)?;
    m.add_function(wrap_pyfunction!(polygamma, m)?)?;
    m.add_function(wrap_pyfunction!(logit, m)?)?;
    m.add_function(wrap_pyfunction!(expit, m)?)?;
    m.add_function(wrap_pyfunction!(sinc, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_j0, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_j1, m)?)?;
    Ok(())
}

/// Helper macro to dispatch special functions
macro_rules! dispatch_special {
    ($input:expr, $method:ident) => {
        match &$input.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let result = t.$method().map_err(|e| {
                    crate::error::convert_error(format!(
                        "special.{} failed: {:?}",
                        stringify!($method),
                        e
                    ))
                })?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(result),
                })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let result = t.$method().map_err(|e| {
                    crate::error::convert_error(format!(
                        "special.{} failed: {:?}",
                        stringify!($method),
                        e
                    ))
                })?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(result),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                let result = t.$method().map_err(|e| {
                    crate::error::convert_error(format!(
                        "special.{} failed: {:?}",
                        stringify!($method),
                        e
                    ))
                })?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(result),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                format!(
                    "special.{} not implemented for sparse tensors",
                    stringify!($method)
                ),
            )),
        }
    };
}

#[pyfunction]
pub fn erf(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, erf)
}

#[pyfunction]
pub fn erfc(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, erfc)
}

#[pyfunction]
pub fn erfinv(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, erfinv)
}

#[pyfunction]
pub fn ndtr(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, ndtr)
}

#[pyfunction]
pub fn gamma(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, gamma)
}

#[pyfunction]
pub fn lgamma(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, lgamma)
}

#[pyfunction]
pub fn digamma(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, digamma)
}

#[pyfunction]
pub fn polygamma(n: usize, input: &PyTensor) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let result = t.polygamma(n).map_err(|e| {
                crate::error::convert_error(format!(
                    "special.polygamma failed: {:?}",
                    e
                ))
            })?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let result = t.polygamma(n).map_err(|e| {
                crate::error::convert_error(format!(
                    "special.polygamma failed: {:?}",
                    e
                ))
            })?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let result = t.polygamma(n).map_err(|e| {
                crate::error::convert_error(format!(
                    "special.polygamma failed: {:?}",
                    e
                ))
            })?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(result),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "special.polygamma not implemented for sparse tensors",
        )),
    }
}

#[pyfunction]
pub fn logit(input: &PyTensor, eps: Option<f64>) -> PyResult<PyTensor> {
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let result = t.logit(eps).map_err(|e| {
                crate::error::convert_error(format!(
                    "special.logit failed: {:?}",
                    e
                ))
            })?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let result = t.logit(eps).map_err(|e| {
                crate::error::convert_error(format!(
                    "special.logit failed: {:?}",
                    e
                ))
            })?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let result = t.logit(eps).map_err(|e| {
                crate::error::convert_error(format!(
                    "special.logit failed: {:?}",
                    e
                ))
            })?;
            Ok(PyTensor {
                inner: TensorWrapper::GpuDenseF32(result),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "special.logit not implemented for sparse tensors",
        )),
    }
}

#[pyfunction]
pub fn expit(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, expit)
}

#[pyfunction]
pub fn sinc(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, sinc)
}

#[pyfunction]
pub fn bessel_j0(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, bessel_j0)
}

#[pyfunction]
pub fn bessel_j1(input: &PyTensor) -> PyResult<PyTensor> {
    dispatch_special!(input, bessel_j1)
}
