use super::{to_py_err, BatchNorm1DWrapper, BatchNorm2DWrapper, BatchNorm3DWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::normalization::{BatchNorm1d, BatchNorm2d, BatchNorm3d};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// BatchNorm1d
// ============================================================================

#[pyclass(name = "BatchNorm1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyBatchNorm1d {
    pub inner: BatchNorm1DWrapper,
}

#[pymethods]
impl PyBatchNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, track_running_stats=true, dtype="float32", device="cpu"))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        track_running_stats: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let momentum_val = momentum.unwrap_or(0.1);
        let track_stats = track_running_stats.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm1DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = BatchNorm1d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new_with_backend(CpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm1DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = BatchNorm1d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(GpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm1DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for BatchNorm1d",
                ))
            }
        };

        Ok(PyBatchNorm1d { inner: wrapper })
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            BatchNorm1DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight.data().clone()),
            },
            BatchNorm1DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight.data().clone()),
            },
            #[cfg(feature = "gpu")]
            BatchNorm1DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight.data().clone()),
            },
        }
    }

    #[getter]
    fn bias(&self) -> PyTensor {
        match &self.inner {
            BatchNorm1DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.bias.data().clone()),
            },
            BatchNorm1DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.bias.data().clone()),
            },
            #[cfg(feature = "gpu")]
            BatchNorm1DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.bias.data().clone()),
            },
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.shape();
        let (output, needs_squeeze) = if input_shape.len() == 2 {
            let unsqueezed = input.unsqueeze(2)?;
            let result = match (&self.inner, &unsqueezed.inner) {
                (BatchNorm1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => m
                    .forward(&i)
                    .map_err(to_py_err)
                    .map(TensorWrapper::CpuDenseF32)?,
                (BatchNorm1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => m
                    .forward(&i)
                    .map_err(to_py_err)
                    .map(TensorWrapper::CpuDenseF64)?,
                #[cfg(feature = "gpu")]
                (BatchNorm1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => m
                    .forward(&i)
                    .map_err(to_py_err)
                    .map(TensorWrapper::GpuDenseF32)?,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input/Module mismatch",
                    ))
                }
            };
            (result, true)
        } else {
            let result = match (&self.inner, &input.inner) {
                (BatchNorm1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => m
                    .forward(&i)
                    .map_err(to_py_err)
                    .map(TensorWrapper::CpuDenseF32)?,
                (BatchNorm1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => m
                    .forward(&i)
                    .map_err(to_py_err)
                    .map(TensorWrapper::CpuDenseF64)?,
                #[cfg(feature = "gpu")]
                (BatchNorm1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => m
                    .forward(&i)
                    .map_err(to_py_err)
                    .map(TensorWrapper::GpuDenseF32)?,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input/Module mismatch",
                    ))
                }
            };
            (result, false)
        };

        if needs_squeeze {
            PyTensor { inner: output }.squeeze(Some(2))
        } else {
            Ok(PyTensor { inner: output })
        }
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            BatchNorm1DWrapper::CpuF32(m) => m.train(mode),
            BatchNorm1DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            BatchNorm1DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![self.weight(), self.bias()]
    }
}

// ============================================================================
// BatchNorm2d
// ============================================================================

#[pyclass(name = "BatchNorm2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyBatchNorm2d {
    pub inner: BatchNorm2DWrapper,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, track_running_stats=true, dtype="float32", device="cpu"))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        track_running_stats: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let momentum_val = momentum.unwrap_or(0.1);
        let track_stats = track_running_stats.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm2DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = BatchNorm2d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new_with_backend(CpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm2DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = BatchNorm2d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(GpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm2DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for BatchNorm2d",
                ))
            }
        };

        Ok(PyBatchNorm2d { inner: wrapper })
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            BatchNorm2DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight.data().clone()),
            },
            BatchNorm2DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight.data().clone()),
            },
            #[cfg(feature = "gpu")]
            BatchNorm2DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight.data().clone()),
            },
        }
    }

    #[getter]
    fn bias(&self) -> PyTensor {
        match &self.inner {
            BatchNorm2DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.bias.data().clone()),
            },
            BatchNorm2DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.bias.data().clone()),
            },
            #[cfg(feature = "gpu")]
            BatchNorm2DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.bias.data().clone()),
            },
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (BatchNorm2DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (BatchNorm2DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (BatchNorm2DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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
            BatchNorm2DWrapper::CpuF32(m) => m.train(mode),
            BatchNorm2DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            BatchNorm2DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![self.weight(), self.bias()]
    }
}

// ============================================================================
// BatchNorm3d
// ============================================================================

#[pyclass(name = "BatchNorm3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyBatchNorm3d {
    pub inner: BatchNorm3DWrapper,
}

#[pymethods]
impl PyBatchNorm3d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, track_running_stats=true, dtype="float32", device="cpu"))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        track_running_stats: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let momentum_val = momentum.unwrap_or(0.1);
        let track_stats = track_running_stats.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm3DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = BatchNorm3d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new_with_backend(CpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm3DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = BatchNorm3d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(GpuBackend::default(), num_features, eps_val, momentum_val, track_stats).map_err(to_py_err)?;
                BatchNorm3DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for BatchNorm3d",
                ))
            }
        };

        Ok(PyBatchNorm3d { inner: wrapper })
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            BatchNorm3DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight.data().clone()),
            },
            BatchNorm3DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight.data().clone()),
            },
            #[cfg(feature = "gpu")]
            BatchNorm3DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight.data().clone()),
            },
        }
    }

    #[getter]
    fn bias(&self) -> PyTensor {
        match &self.inner {
            BatchNorm3DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.bias.data().clone()),
            },
            BatchNorm3DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.bias.data().clone()),
            },
            #[cfg(feature = "gpu")]
            BatchNorm3DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.bias.data().clone()),
            },
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (BatchNorm3DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (BatchNorm3DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (BatchNorm3DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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
            BatchNorm3DWrapper::CpuF32(m) => m.train(mode),
            BatchNorm3DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            BatchNorm3DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![self.weight(), self.bias()]
    }
}
