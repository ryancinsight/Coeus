use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::utility::Flatten;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use backend::CpuBackend;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

#[derive(Clone)]
pub enum FlattenWrapper {
    CpuF32(Flatten<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Flatten<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Flatten<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "Flatten", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyFlatten {
    pub inner: FlattenWrapper,
    pub start_dim: isize,
    pub end_dim: isize,
}

#[pymethods]
impl PyFlatten {
    #[new]
    #[pyo3(signature = (start_dim=1, end_dim=-1))]
    fn new(start_dim: isize, end_dim: isize) -> Self {
        // We defer creation of inner wrapper until we know the type, or simpler:
        // just create one variant, as Flatten parameters (dims) are stored in struct anyway?
        // Wait, Flatten struct DOES store start/end dim.
        // And `Module` trait forward needs type match.
        
        // Let's presume generic default. Ideally we should match input type on forward, 
        // but Module instance usually has fixed backend/type. 
        // PyTorch modules adapt to input type but stored buffers might not.
        // Flatten has no buffers/parameters.
        
        // I will initialize default F32 wrapper but allow dynamic dispatch if I can switching wrappers? 
        // No, `inner` is fixed.
        // But `Flatten` is stateless except for dims. I can recreate it in forward if needed?
        // Or just store the wrapper and use match.
        
        let flatten = Flatten::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            start_dim, 
            0, // Placeholder
        );
        
        PyFlatten {
            inner: FlattenWrapper::CpuF32(flatten),
            start_dim,
            end_dim,
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Calculate actual dims
        let ndim = input.shape().len() as isize;
        let start = if self.start_dim < 0 { ndim + self.start_dim } else { self.start_dim };
        let end = if self.end_dim < 0 { ndim + self.end_dim } else { self.end_dim };
        
        if start < 0 || start >= ndim || end < 0 || end >= ndim {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid dimensions for Flatten: start={}, end={}, ndim={}", self.start_dim, self.end_dim, ndim)
            ));
        }

        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let module = Flatten::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(start, end);
                let res = module.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(i) => {
                 let module = Flatten::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(start, end);
                let res = module.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                 let module = Flatten::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(start, end);
                let res = module.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
             _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for Flatten",
            )),
        }
    }
}
