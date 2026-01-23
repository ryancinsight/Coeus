use super::{to_py_err, RMSNormWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::normalization::RMSNorm;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// RMSNorm
// ============================================================================

#[pyclass(name = "RMSNorm", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyRMSNorm {
    pub inner: RMSNormWrapper,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5, dtype="float32", device="cpu"))]
    fn new(
        normalized_shape: Bound<'_, PyAny>,
        eps: Option<f64>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let shape: Vec<usize> = if let Ok(s) = normalized_shape.extract::<usize>() {
            vec![s]
        } else {
            normalized_shape.extract::<Vec<usize>>()?
        };

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = RMSNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    shape, eps_val, true,
                );
                RMSNormWrapper::CpuF32(m.map_err(to_py_err)?)
            }
            ("cpu", "float64") => {
                let m = RMSNorm::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    shape, eps_val, true,
                );
                RMSNormWrapper::CpuF64(m.map_err(to_py_err)?)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = RMSNorm::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    shape, eps_val, true,
                );
                RMSNormWrapper::GpuF32(m.map_err(to_py_err)?)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for RMSNorm",
                ))
            }
        };

        Ok(PyRMSNorm { inner: wrapper })
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            RMSNormWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight().data().clone()),
            },
            RMSNormWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight().data().clone()),
            },
            #[cfg(feature = "gpu")]
            RMSNormWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight().data().clone()),
            },
        }
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            RMSNormWrapper::CpuF32(m) => m.train(mode),
            RMSNormWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            RMSNormWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (RMSNormWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (RMSNormWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (RMSNormWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input/Module backend/dtype mismatch",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![self.weight()]
    }
}
