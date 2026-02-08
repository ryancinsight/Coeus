use super::class::PyTensor;
use crate::error;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};
use ::std::vec::Vec;

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(masked_select, m)?)?;
    // where is keyword so we use alias
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_std, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_var, m)?)?;
    
    // Linalg aliases
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(bmm, m)?)?;
    m.add_function(wrap_pyfunction!(addmm, m)?)?;
    m.add_function(wrap_pyfunction!(addmv, m)?)?;
    m.add_function(wrap_pyfunction!(addbmm, m)?)?;
    m.add_function(wrap_pyfunction!(baddbmm, m)?)?;
    m.add_function(wrap_pyfunction!(addr, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;

    // Comparison/Logical Ops
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(logical_not, m)?)?;

    // Math Ops
    m.add_function(wrap_pyfunction!(atan2, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(rsqrt, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(erfinv, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (input, mask))]
pub fn masked_select(input: &PyTensor, mask: &PyTensor) -> PyResult<PyTensor> {
    crate::tensor::ops::indexing::masked_select(input, mask)
}

#[pyfunction(name = "where")]
#[pyo3(signature = (condition, input, other))]
pub fn where_(condition: &PyTensor, input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    crate::tensor::ops::comparison::where_(condition, input, other)
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
pub fn argmax(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let dim_usize = dim.map(|d| {
         if d < 0 { (input.shape().len() as isize + d) as usize } else { d as usize }
    });
    input.argmax(dim_usize, keepdim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmin(input: &PyTensor, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let dim_usize = dim.map(|d| {
         if d < 0 { (input.shape().len() as isize + d) as usize } else { d as usize }
    });
    input.argmin(dim_usize, keepdim)
}

#[pyfunction(name = "std")]
#[pyo3(signature = (input, dim=None, correction=1, keepdim=false))]
pub fn tensor_std(input: &PyTensor, dim: Option<isize>, correction: usize, keepdim: bool) -> PyResult<PyTensor> {
    let dim_usize = dim.map(|d| {
         if d < 0 { (input.shape().len() as isize + d) as usize } else { d as usize }
    });
    input.std(dim_usize, correction, keepdim)
}

#[pyfunction(name = "var")]
#[pyo3(signature = (input, dim=None, correction=1, keepdim=false))]
pub fn tensor_var(input: &PyTensor, dim: Option<isize>, correction: usize, keepdim: bool) -> PyResult<PyTensor> {
    let dim_usize = dim.map(|d| {
         if d < 0 { (input.shape().len() as isize + d) as usize } else { d as usize }
    });
    input.var(dim_usize, correction, keepdim)
}

#[pyfunction]
pub fn matmul(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.matmul(other)
}

#[pyfunction]
pub fn bmm(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.bmm(other)
}

#[pyfunction]
#[pyo3(signature = (input, mat1, mat2, *, beta=1.0, alpha=1.0))]
pub fn addmm(input: &PyTensor, mat1: &PyTensor, mat2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
    input.addmm(mat1, mat2, beta, alpha)
}

#[pyfunction]
#[pyo3(signature = (input, mat, vec, *, beta=1.0, alpha=1.0))]
pub fn addmv(input: &PyTensor, mat: &PyTensor, vec: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
    input.addmv(mat, vec, beta, alpha)
}

#[pyfunction]
#[pyo3(signature = (input, batch1, batch2, *, beta=1.0, alpha=1.0))]
pub fn addbmm(input: &PyTensor, batch1: &PyTensor, batch2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
    input.addbmm(batch1, batch2, beta, alpha)
}

#[pyfunction]
#[pyo3(signature = (input, batch1, batch2, *, beta=1.0, alpha=1.0))]
pub fn baddbmm(input: &PyTensor, batch1: &PyTensor, batch2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
    input.baddbmm(batch1, batch2, beta, alpha)
}

#[pyfunction]
#[pyo3(signature = (input, vec1, vec2, *, beta=1.0, alpha=1.0))]
pub fn addr(input: &PyTensor, vec1: &PyTensor, vec2: &PyTensor, beta: f64, alpha: f64) -> PyResult<PyTensor> {
    input.addr(vec1, vec2, beta, alpha)
}

#[pyfunction]
pub fn outer(input: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
    // Note: outer is vec1 (input) @ vec2^T
    input.outer(vec2)
}

#[pyfunction]
pub fn isnan(input: &PyTensor) -> PyResult<PyTensor> {
    input.isnan()
}

#[pyfunction]
pub fn isinf(input: &PyTensor) -> PyResult<PyTensor> {
    input.isinf()
}

#[pyfunction]
pub fn isfinite(input: &PyTensor) -> PyResult<PyTensor> {
    input.isfinite()
}

#[pyfunction]
pub fn logical_and(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.logical_and(other)
}

#[pyfunction]
pub fn logical_or(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.logical_or(other)
}

#[pyfunction]
pub fn logical_xor(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.logical_xor(other)
}

#[pyfunction]
pub fn logical_not(input: &PyTensor) -> PyResult<PyTensor> {
    input.logical_not()
}

#[pyfunction]
pub fn atan2(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.atan2(other)
}

#[pyfunction]
pub fn sqrt(input: &PyTensor) -> PyResult<PyTensor> {
    input.sqrt()
}

#[pyfunction]
pub fn rsqrt(input: &PyTensor) -> PyResult<PyTensor> {
    input.rsqrt()
}

#[pyfunction]
pub fn erf(input: &PyTensor) -> PyResult<PyTensor> {
    input.erf()
}

#[pyfunction]
pub fn erfc(input: &PyTensor) -> PyResult<PyTensor> {
    input.erfc()
}

#[pyfunction]
pub fn erfinv(input: &PyTensor) -> PyResult<PyTensor> {
    input.erfinv()
}

#[pyfunction]
pub fn log1p(input: &PyTensor) -> PyResult<PyTensor> {
    input.log1p()
}

#[pyfunction]
pub fn expm1(input: &PyTensor) -> PyResult<PyTensor> {
    input.expm1()
}

#[pyfunction]
pub fn reciprocal(input: &PyTensor) -> PyResult<PyTensor> {
    input.reciprocal()
}
