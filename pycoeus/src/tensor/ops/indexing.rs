use crate::tensor::class::{PyTensor, to_py_err, TensorWrapper};
pub use crate::tensor::wrapper::WrapTensor;
use crate::dispatch_tensor;
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pymethods]
impl PyTensor {
    pub fn gather(&self, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
        gather(self, dim, index)
    }

    pub fn index_select(&self, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
        index_select(self, dim, index)
    }

    pub fn take(&self, indices: &PyTensor) -> PyResult<PyTensor> {
        take(self, indices)
    }

    pub fn put(&mut self, indices: &PyTensor, values: &PyTensor, accumulate: bool) -> PyResult<PyTensor> {
        put(self, indices, values, accumulate)
    }

    pub fn masked_select(&self, mask: &PyTensor) -> PyResult<PyTensor> {
        masked_select(self, mask)
    }

    pub fn masked_fill(&self, mask: &PyTensor, value: f64) -> PyResult<PyTensor> {
        masked_fill(self, mask, value)
    }

    pub fn nonzero(&self) -> PyResult<PyTensor> {
        nonzero(self)
    }

    pub fn index_add(&self, dim: usize, index: &PyTensor, source: &PyTensor, alpha: Option<f64>) -> PyResult<PyTensor> {
        index_add(self, dim, index, source, alpha.unwrap_or(1.0))
    }

    pub fn index_add_(&mut self, dim: usize, index: &PyTensor, source: &PyTensor, alpha: Option<f64>) -> PyResult<PyTensor> {
        index_add_(self, dim, index, source, alpha.unwrap_or(1.0))?;
        Ok(self.clone())
    }
}

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

pub fn masked_select(tensor: &PyTensor, mask: &PyTensor) -> PyResult<PyTensor> {
    dispatch_tensor!(tensor, t => {
        dispatch_tensor!(mask, m => {
             // Cast to U8 specifically for mask
             let m_u8 = ::tensor::ops::cast::cast::<dtype::int::UInt8, _, _, _>(m).map_err(to_py_err)?;
             let res = ::tensor::ops::indexing::masked::masked_select(t, &m_u8).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    })
}

trait FromF64 {
    fn from_f64(v: f64) -> Self;
}

impl FromF64 for dtype::float::Float32 { fn from_f64(v: f64) -> Self { Self::new(v as f32) } }
impl FromF64 for dtype::float::Float64 { fn from_f64(v: f64) -> Self { Self::new(v) } }
impl FromF64 for dtype::int::Int64 { fn from_f64(v: f64) -> Self { Self::new(v as i64) } }
impl FromF64 for dtype::complex::Complex32 { fn from_f64(v: f64) -> Self { Self::new(v as f32, 0.0) } }

pub fn masked_fill(tensor: &PyTensor, mask: &PyTensor, value: f64) -> PyResult<PyTensor> {
    let dense = tensor.contiguous()?;
    dispatch_tensor!(dense, t => {
        dispatch_tensor!(mask, m => {
             let m_u8 = ::tensor::ops::cast::cast::<dtype::int::UInt8, _, _, _>(m).map_err(to_py_err)?;
             let val_t = FromF64::from_f64(value);
             let res = ::tensor::ops::indexing::masked::masked_fill(t, &m_u8, val_t).map_err(to_py_err)?;
             Ok(PyTensor { inner: res.wrap() })
        })
    })
}

pub fn nonzero(tensor: &PyTensor) -> PyResult<PyTensor> {
    dispatch_tensor!(tensor, t => {
        let res = ::tensor::ops::indexing::nonzero(t).map_err(to_py_err)?;
        Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
    })
}

pub fn index_add(tensor: &PyTensor, dim: usize, index: &PyTensor, source: &PyTensor, alpha: f64) -> PyResult<PyTensor> {
    let mut cloned = tensor.clone();
    index_add_(&mut cloned, dim, index, source, alpha)?;
    Ok(cloned)
}

pub fn index_add_(tensor: &mut PyTensor, dim: usize, index: &PyTensor, source: &PyTensor, alpha: f64) -> PyResult<()> {
    // Extract index as I64 tensor
    let idx = match &index.inner {
        TensorWrapper::CpuDenseI64(i) => i,
        TensorWrapper::CpuStridedI64(i) => {
            // If strided, index_add implementation needs to handle it or we cast to dense
            // index_add currently uses to_dense_generic internally if needed.
            // But the signature requires S2: StorageToDense<I64> + StorageFromVec<I64>.
            // StridedStorage implements StorageToDense but NOT StorageFromVec (creation).
            // Actually, index_add signature has S2: StorageFromVec<I64>.
            // I'll cast index to dense if it's strided.
            return Err(to_py_err("index_add: index must be dense I64 (strided indices not yet supported for index_add)"));
        }
        _ => return Err(to_py_err("index_add: index must be I64")),
    };

    match (&mut tensor.inner, &source.inner) {
        (TensorWrapper::CpuDenseF32(t), TensorWrapper::CpuDenseF32(s)) => {
            ::tensor::ops::indexing::index_add(t, dim, idx, s, dtype::float::Float32::new(alpha as f32)).map_err(to_py_err)?;
        },
        (TensorWrapper::CpuDenseF64(t), TensorWrapper::CpuDenseF64(s)) => {
            ::tensor::ops::indexing::index_add(t, dim, idx, s, dtype::float::Float64::new(alpha)).map_err(to_py_err)?;
        },
        (TensorWrapper::CpuStridedF32(t), TensorWrapper::CpuStridedF32(s)) => {
            ::tensor::ops::indexing::index_add(t, dim, idx, s, dtype::float::Float32::new(alpha as f32)).map_err(to_py_err)?;
        },
        (TensorWrapper::CpuStridedF64(t), TensorWrapper::CpuStridedF64(s)) => {
            ::tensor::ops::indexing::index_add(t, dim, idx, s, dtype::float::Float64::new(alpha)).map_err(to_py_err)?;
        },
        _ => return Err(to_py_err("index_add: type/storage mismatch or unsupported")),
    }
    Ok(())
}


