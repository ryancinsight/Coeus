use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::modules::utility::Identity;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use backend::CpuBackend;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

#[derive(Clone)]
pub enum IdentityWrapper {
    CpuF32(Identity<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Identity<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Identity<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "Identity", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyIdentity {
    pub inner: IdentityWrapper,
}

#[pymethods]
impl PyIdentity {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        // Default to F32 for now, or match input on forward?
        // Module usually holds type state if parameters exist, but Identity has none.
        // We can just picking one variant, it doesn't matter much for Identity.
        let identity = Identity::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        PyIdentity {
            inner: IdentityWrapper::CpuF32(identity),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Identity just returns input clone
        Ok(PyTensor {
            inner: input.inner.clone(),
        })
    }
}
