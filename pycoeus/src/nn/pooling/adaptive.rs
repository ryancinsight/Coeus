use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
};
use pyo3::prelude::*;

#[pyclass(name = "AdaptiveAvgPool1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveAvgPool1d {
    pub inner: AdaptiveAvgPool1d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAdaptiveAvgPool1d {
    #[new]
    #[pyo3(signature = (output_size, dtype="float32", device="cpu"))]
    fn new(output_size: usize, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let pool = AdaptiveAvgPool1d::new(output_size);
        Ok(PyAdaptiveAvgPool1d {
            inner: pool,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype not supported for AdaptiveAvgPool1d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AdaptiveAvgPool2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveAvgPool2d {
    pub inner: AdaptiveAvgPool2d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    #[pyo3(signature = (output_size, dtype="float32", device="cpu"))]
    fn new(
        output_size: Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            (s, s)
        } else {
            output_size.extract::<(usize, usize)>()?
        };

        let pool = AdaptiveAvgPool2d::new(size);
        Ok(PyAdaptiveAvgPool2d {
            inner: pool,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype not supported for AdaptiveAvgPool2d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AdaptiveAvgPool3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveAvgPool3d {
    pub inner: AdaptiveAvgPool3d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAdaptiveAvgPool3d {
    #[new]
    #[pyo3(signature = (output_size, dtype="float32", device="cpu"))]
    fn new(
        output_size: Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            (s, s, s)
        } else {
            output_size.extract::<(usize, usize, usize)>()?
        };

        let pool = AdaptiveAvgPool3d::new(size);
        Ok(PyAdaptiveAvgPool3d {
            inner: pool,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype not supported for AdaptiveAvgPool3d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AdaptiveMaxPool1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveMaxPool1d {
    pub inner: AdaptiveMaxPool1d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAdaptiveMaxPool1d {
    #[new]
    #[pyo3(signature = (output_size, dtype="float32", device="cpu"))]
    fn new(output_size: usize, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let pool = AdaptiveMaxPool1d::new(output_size);
        Ok(PyAdaptiveMaxPool1d {
            inner: pool,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype not supported for AdaptiveMaxPool1d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AdaptiveMaxPool2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveMaxPool2d {
    pub inner: AdaptiveMaxPool2d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAdaptiveMaxPool2d {
    #[new]
    #[pyo3(signature = (output_size, dtype="float32", device="cpu"))]
    fn new(
        output_size: Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            (s, s)
        } else {
            output_size.extract::<(usize, usize)>()?
        };

        let pool = AdaptiveMaxPool2d::new(size);
        Ok(PyAdaptiveMaxPool2d {
            inner: pool,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype not supported for AdaptiveMaxPool2d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AdaptiveMaxPool3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveMaxPool3d {
    pub inner: AdaptiveMaxPool3d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAdaptiveMaxPool3d {
    #[new]
    #[pyo3(signature = (output_size, dtype="float32", device="cpu"))]
    fn new(
        output_size: Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            (s, s, s)
        } else {
            output_size.extract::<(usize, usize, usize)>()?
        };

        let pool = AdaptiveMaxPool3d::new(size);
        Ok(PyAdaptiveMaxPool3d {
            inner: pool,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = self.inner.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype not supported for AdaptiveMaxPool3d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}
