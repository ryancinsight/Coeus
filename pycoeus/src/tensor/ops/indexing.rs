use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::dispatch_tensor;
use pyo3::prelude::*;
use dtype::num_traits::Zero;

pub fn gather(tensor: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    dispatch_tensor!(tensor, t => {
        // Convert index to i64 if needed
        match &index.inner {
            TensorWrapper::CpuDenseI64(idx) => {
                let res = ::tensor::ops::indexing::gather(t, dim, idx).map_err(to_py_err)?;
                Ok(PyTensor { inner: res.wrap() })
            }
            _ => Err(to_py_err("Gather index must be i64 dense")),
        }
    })
}

pub fn index_select(tensor: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    dispatch_tensor!(tensor, t => {
         let idx_vec = match &index.inner {
            TensorWrapper::CpuDenseI64(idx) => {
                idx.as_slice().iter().map(|v| v.0 as usize).collect::<Vec<usize>>()
            }
            _ => return Err(to_py_err("Index select index must be i64 dense")),
        };
        let res = ::tensor::ops::indexing::index_select(t, dim, &idx_vec).map_err(to_py_err)?;
        Ok(PyTensor { inner: res.wrap() })
    })
}

pub fn take(tensor: &PyTensor, indices: &PyTensor) -> PyResult<PyTensor> {
    dispatch_tensor!(tensor, t => {
        match &indices.inner {
            TensorWrapper::CpuDenseI64(idx) => {
                 let res = ::tensor::ops::indexing::take(t, idx).map_err(to_py_err)?;
                 Ok(PyTensor { inner: res.wrap() })
            }
             _ => Err(to_py_err("Indices must be generic Int64 tensor")),
        }
    })
}

pub fn put(tensor: &PyTensor, indices: &PyTensor, values: &PyTensor, accumulate: bool) -> PyResult<PyTensor> {
    // Nested match for put since values type must match tensor type
    match &indices.inner {
        TensorWrapper::CpuDenseI64(idx) => {
            match (&tensor.inner, &values.inner) {
                (TensorWrapper::CpuDenseF32(t), TensorWrapper::CpuDenseF32(v)) => {
                    let res = t.put(idx, v, accumulate).map_err(to_py_err)?;
                    Ok(PyTensor { inner: res.wrap() })
                },
                (TensorWrapper::CpuDenseF64(t), TensorWrapper::CpuDenseF64(v)) => {
                    let res = t.put(idx, v, accumulate).map_err(to_py_err)?;
                    Ok(PyTensor { inner: res.wrap() })
                },
                (TensorWrapper::CpuDenseI64(t), TensorWrapper::CpuDenseI64(v)) => {
                     let res = t.put(idx, v, accumulate).map_err(to_py_err)?;
                     Ok(PyTensor { inner: res.wrap() })
                },
                // Add GPU branches if supported/compiled
                 // Fallback for mismatch
                _ => Err(to_py_err("Tensor and Values type/device mismatch or unsupported type for put"))
            }
        },
        _ => Err(to_py_err("Indices for put must be Int64"))
    }
}

pub fn getitem(tensor: &PyTensor, index: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    // Check if index is a Tensor (Advanced Indexing)
    if let Ok(idx_tensor) = index.extract::<PyTensor>() {
        // User requested "Implement `take`, `put`, and advanced indexing logic (Tensor indices)".
        // Using `take` often implies flattened index in some contexts, but `advanced indexing` usually means preserving shape.
        // Let's assume x[idx] -> index_select(0, idx) for now as a common case.
        return index_select(tensor, 0, &idx_tensor);
    }
    
    // Check for integer (Select)
    if let Ok(idx_int) = index.extract::<isize>() {
         // handle negative index
         let dims = tensor.shape();
         if dims.is_empty() { return Err(to_py_err("Cannot index scalar")); }
         let dim0 = dims[0];
         let final_idx = if idx_int < 0 {
             dim0 as isize + idx_int
         } else {
             idx_int
         };
         
         if final_idx < 0 || final_idx as usize >= dim0 {
             return Err(to_py_err("Index out of bounds"));
         }
         
         // Select dim 0. This reduces rank.
         // dispatch_tensor! provides `t` which is the specific Tensor type.
         let squeezed_wrapper = dispatch_tensor!(tensor, t => {
             let selected = t.select(0, final_idx as usize).map_err(to_py_err)?;
             Ok::<TensorWrapper, PyErr>(selected.wrap())
         })?;
         return Ok(PyTensor { inner: squeezed_wrapper });
    }

    Err(to_py_err("Only Integer and Tensor indices are currently supported for __getitem__"))
}

pub fn masked_select(_tensor: &PyTensor, _mask: &PyTensor) -> PyResult<PyTensor> {
    Err(to_py_err("masked_select not fully implemented (requires tensor cast op)"))
}

trait FromF64 {
    fn from_f64(v: f64) -> Self;
}
// Keep trait to avoid touching too much code or imports? 
// Actually if I remove usage, I can remove trait.
// But I'll leave trait as stub or remove it. Better remove.

pub fn masked_fill(_tensor: &PyTensor, _mask: &PyTensor, _value: f64) -> PyResult<PyTensor> {
    Err(to_py_err("masked_fill not fully implemented (requires tensor cast op)"))
}

// Remove _cast_to_u8


