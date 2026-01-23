use super::{to_py_err, GroupNormWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::normalization::GroupNorm;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// GroupNorm
// ============================================================================

#[pyclass(name = "GroupNorm", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyGroupNorm {
    pub inner: GroupNormWrapper,
}

#[pymethods]
impl PyGroupNorm {
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps=1e-5, affine=true, dtype="float32", device="cpu"))]
    fn new(
        num_groups: usize,
        num_channels: usize,
        eps: Option<f64>,
        affine: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let affine_val = affine.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_groups,
                    num_channels,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                GroupNormWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = GroupNorm::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    num_groups,
                    num_channels,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                GroupNormWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = GroupNorm::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_groups,
                    num_channels,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                GroupNormWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for GroupNorm",
                ))
            }
        };

        Ok(PyGroupNorm { inner: wrapper })
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            GroupNormWrapper::CpuF32(m) => m.train(mode),
            GroupNormWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            GroupNormWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (GroupNormWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (GroupNormWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (GroupNormWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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
        let mut params = Vec::new();
        match &self.inner {
            GroupNormWrapper::CpuF32(m) => {
                if m.affine {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF32(m.weight.data().clone()),
                    });
                }
                if m.affine {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF32(m.bias.data().clone()),
                    });
                }
            }
            GroupNormWrapper::CpuF64(m) => {
                if m.affine {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF64(m.weight.data().clone()),
                    });
                }
                if m.affine {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF64(m.bias.data().clone()),
                    });
                }
            }
            #[cfg(feature = "gpu")]
            GroupNormWrapper::GpuF32(m) => {
                if m.affine {
                    params.push(PyTensor {
                        inner: TensorWrapper::GpuDenseF32(m.weight.data().clone()),
                    });
                }
                if m.affine {
                    params.push(PyTensor {
                        inner: TensorWrapper::GpuDenseF32(m.bias.data().clone()),
                    });
                }
            }
        }
        params
    }
}
