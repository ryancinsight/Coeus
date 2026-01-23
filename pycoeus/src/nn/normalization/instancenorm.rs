use super::{to_py_err, InstanceNorm1DWrapper, InstanceNorm2DWrapper, InstanceNorm3DWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::normalization::InstanceNorm;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// InstanceNorm1d
// ============================================================================

/// Instance Normalization 1D.
///
/// Applies Instance Normalization over a 1D input.
/// This is equivalent to GroupNorm with num_groups = num_channels.
#[pyclass(name = "InstanceNorm1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyInstanceNorm1d {
    pub inner: InstanceNorm1DWrapper,
}

#[pymethods]
impl PyInstanceNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, affine=true, dtype="float32", device="cpu"))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let _momentum_val = momentum.unwrap_or(0.1);
        let affine_val = affine.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm1DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = InstanceNorm::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm1DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = InstanceNorm::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm1DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for InstanceNorm1d",
                ))
            }
        };

        Ok(PyInstanceNorm1d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (InstanceNorm1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (InstanceNorm1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (InstanceNorm1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            InstanceNorm1DWrapper::CpuF32(m) => m.train(mode),
            InstanceNorm1DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            InstanceNorm1DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        match &self.inner {
            InstanceNorm1DWrapper::CpuF32(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(bias.data().clone()),
                });
            }
            InstanceNorm1DWrapper::CpuF64(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(bias.data().clone()),
                });
            }
            #[cfg(feature = "gpu")]
            InstanceNorm1DWrapper::GpuF32(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(bias.data().clone()),
                });
            }
        }
        params
    }
}

// ============================================================================
// InstanceNorm2d
// ============================================================================

/// Instance Normalization 2D.
///
/// Applies Instance Normalization over a 2D input.
/// This is equivalent to GroupNorm with num_groups = num_channels.
#[pyclass(name = "InstanceNorm2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyInstanceNorm2d {
    pub inner: InstanceNorm2DWrapper,
}

#[pymethods]
impl PyInstanceNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, affine=true, dtype="float32", device="cpu"))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let _momentum_val = momentum.unwrap_or(0.1);
        let affine_val = affine.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm2DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = InstanceNorm::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm2DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = InstanceNorm::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm2DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for InstanceNorm2d",
                ))
            }
        };

        Ok(PyInstanceNorm2d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (InstanceNorm2DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (InstanceNorm2DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (InstanceNorm2DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            InstanceNorm2DWrapper::CpuF32(m) => m.train(mode),
            InstanceNorm2DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            InstanceNorm2DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        match &self.inner {
            InstanceNorm2DWrapper::CpuF32(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(bias.data().clone()),
                });
            }
            InstanceNorm2DWrapper::CpuF64(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(bias.data().clone()),
                });
            }
            #[cfg(feature = "gpu")]
            InstanceNorm2DWrapper::GpuF32(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(bias.data().clone()),
                });
            }
        }
        params
    }
}

// ============================================================================
// InstanceNorm3d
// ============================================================================

/// Instance Normalization 3D.
///
/// Applies Instance Normalization over a 3D input.
/// This is equivalent to GroupNorm with num_groups = num_channels.
#[pyclass(name = "InstanceNorm3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyInstanceNorm3d {
    pub inner: InstanceNorm3DWrapper,
}

#[pymethods]
impl PyInstanceNorm3d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, affine=true, dtype="float32", device="cpu"))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let _momentum_val = momentum.unwrap_or(0.1);
        let affine_val = affine.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm3DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = InstanceNorm::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm3DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = InstanceNorm::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_features,
                    eps_val,
                    affine_val,
                )
                .map_err(to_py_err)?;
                InstanceNorm3DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for InstanceNorm3d",
                ))
            }
        };

        Ok(PyInstanceNorm3d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (InstanceNorm3DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (InstanceNorm3DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (InstanceNorm3DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            InstanceNorm3DWrapper::CpuF32(m) => m.train(mode),
            InstanceNorm3DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            InstanceNorm3DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        match &self.inner {
            InstanceNorm3DWrapper::CpuF32(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(bias.data().clone()),
                });
            }
            InstanceNorm3DWrapper::CpuF64(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(bias.data().clone()),
                });
            }
            #[cfg(feature = "gpu")]
            InstanceNorm3DWrapper::GpuF32(m) => {
                let weight = m.weight();
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(weight.data().clone()),
                });
                let bias = m.bias();
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(bias.data().clone()),
                });
            }
        }
        params
    }
}
