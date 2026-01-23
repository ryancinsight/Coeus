use super::class::PyTensor;
use crate::error;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};
use std::vec::Vec;

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    Ok(())
}

/// Concatenate tensors along an existing dimension
#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn cat(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    use crate::tensor::TensorWrapper;

    if tensors.is_empty() {
        return Err(error::convert_error("Cannot concatenate empty tensor list"));
    }

    // Check all tensors have the same type and concatenate based on first tensor's type
    match &tensors[0].inner {
        TensorWrapper::CpuDenseF32(first) => {
            let mut all_tensors = vec![first.clone()];
            for t in tensors.iter().skip(1) {
                if let TensorWrapper::CpuDenseF32(inner) = &t.inner {
                    all_tensors.push(inner.clone());
                } else {
                    return Err(error::convert_error("All tensors must have the same dtype"));
                }
            }
            let result = ::tensor::ops::tensor_ops::concatenate_tensors(&all_tensors, dim)
                .map_err(|e| error::convert_error_with_context(e, "cat"))?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(first) => {
            let mut all_tensors = vec![first.clone()];
            for t in tensors.iter().skip(1) {
                if let TensorWrapper::CpuDenseF64(inner) = &t.inner {
                    all_tensors.push(inner.clone());
                } else {
                    return Err(error::convert_error("All tensors must have the same dtype"));
                }
            }
            let result = ::tensor::ops::tensor_ops::concatenate_tensors(&all_tensors, dim)
                .map_err(|e| error::convert_error_with_context(e, "cat"))?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        _ => Err(error::convert_error(
            "cat not implemented for this tensor type",
        )),
    }
}

/// Stack tensors along a new dimension
#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn stack(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    use crate::tensor::TensorWrapper;

    if tensors.is_empty() {
        return Err(error::convert_error("Cannot stack empty tensor list"));
    }

    // Check all tensors have the same type and stack based on first tensor's type
    match &tensors[0].inner {
        TensorWrapper::CpuDenseF32(first) => {
            let mut all_tensors = vec![first.clone()];
            for t in tensors.iter().skip(1) {
                if let TensorWrapper::CpuDenseF32(inner) = &t.inner {
                    all_tensors.push(inner.clone());
                } else {
                    return Err(error::convert_error("All tensors must have the same dtype"));
                }
            }
            let result = ::tensor::ops::tensor_ops::stack_tensors(&all_tensors, dim)
                .map_err(|e| error::convert_error_with_context(e, "stack"))?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        TensorWrapper::CpuDenseF64(first) => {
            let mut all_tensors = vec![first.clone()];
            for t in tensors.iter().skip(1) {
                if let TensorWrapper::CpuDenseF64(inner) = &t.inner {
                    all_tensors.push(inner.clone());
                } else {
                    return Err(error::convert_error("All tensors must have the same dtype"));
                }
            }
            let result = ::tensor::ops::tensor_ops::stack_tensors(&all_tensors, dim)
                .map_err(|e| error::convert_error_with_context(e, "stack"))?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        _ => Err(error::convert_error(
            "stack not implemented for this tensor type",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmax(input: &PyTensor, dim: Option<i64>, keepdim: bool) -> PyResult<PyTensor> {
    let dim_usize = dim.map(|d| d as usize);
    input.argmax(dim_usize, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmin(input: &PyTensor, dim: Option<i64>, keepdim: bool) -> PyResult<PyTensor> {
    let dim_usize = dim.map(|d| d as usize);
    input.argmin(dim_usize, keepdim)
}
