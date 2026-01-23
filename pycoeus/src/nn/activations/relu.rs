use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::activation::{Hardtanh, LeakyReLU, PReLU, ReLU}; // Hardtanh used in ReLU6
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============ ReLU ============
#[pyclass(name = "ReLU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyReLU;

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self {
        PyReLU
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = ReLU::new().forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = ReLU::new().forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = ReLU::new().forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for ReLU",
            )),
        }
    }
}

// ============ ReLU6 ============
#[pyclass(name = "ReLU6", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyReLU6;

#[pymethods]
impl PyReLU6 {
    #[new]
    fn new() -> Self {
        PyReLU6
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // ReLU6 = min(max(0, x), 6)
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = Hardtanh::new(Float32::new(0.0), Float32::new(6.0))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = Hardtanh::new(Float64::new(0.0), Float64::new(6.0))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = Hardtanh::new(Float32::new(0.0), Float32::new(6.0))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type",
            )),
        }
    }
}

// ============ LeakyReLU ============
#[pyclass(name = "LeakyReLU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLeakyReLU {
    pub negative_slope: f64,
}

#[pymethods]
impl PyLeakyReLU {
    #[new]
    #[pyo3(signature = (negative_slope=0.01))]
    fn new(negative_slope: f64) -> Self {
        PyLeakyReLU { negative_slope }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = LeakyReLU::new(Float32::new(self.negative_slope as f32))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = LeakyReLU::new(Float64::new(self.negative_slope))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = LeakyReLU::new(Float32::new(self.negative_slope as f32))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for LeakyReLU",
            )),
        }
    }
}

// ============ PReLU ============
#[derive(Clone)]
pub enum PReLUWrapper {
    CpuF32(PReLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(PReLU<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(PReLU<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "PReLU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyPReLU {
    pub inner: PReLUWrapper,
}

#[pymethods]
impl PyPReLU {
    #[new]
    #[pyo3(signature = (num_parameters=1, init=0.25, dtype="float32", device="cpu"))]
    fn new(
        num_parameters: usize,
        init: f32,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_parameters,
                    Some(Float32::new(init)),
                );
                PReLUWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = PReLU::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    num_parameters,
                    Some(Float64::new(init as f64)),
                );
                PReLUWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = PReLU::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    num_parameters,
                    Some(Float32::new(init)),
                );
                PReLUWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for PReLU",
                ))
            }
        };

        Ok(PyPReLU { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (PReLUWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (PReLUWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (PReLUWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            PReLUWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight.data().clone()),
            },
            PReLUWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight.data().clone()),
            },
            #[cfg(feature = "gpu")]
            PReLUWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight.data().clone()),
            },
        }
    }
}
