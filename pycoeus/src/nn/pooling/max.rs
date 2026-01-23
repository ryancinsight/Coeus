use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::pooling::{MaxPool1d, MaxPool2d, MaxPool3d};
use pyo3::prelude::*;

#[pyclass(name = "MaxPool1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMaxPool1d {
    pub inner: MaxPool1d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let p = padding.unwrap_or(0);
        let pool = MaxPool1d::new(kernel_size, stride, p);
        Ok(PyMaxPool1d {
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
                "Input backend/dtype not supported for MaxPool1d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "MaxPool2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMaxPool2d {
    pub inner: MaxPool2d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let k = if let Ok(s) = kernel_size.extract::<usize>() {
            (s, s)
        } else {
            kernel_size.extract::<(usize, usize)>()?
        };

        let s = if let Some(st) = stride {
            if let Ok(s_val) = st.extract::<usize>() {
                Some((s_val, s_val))
            } else {
                Some(st.extract::<(usize, usize)>()?)
            }
        } else {
            None
        };

        let p = if let Some(pad) = padding {
            if let Ok(p_val) = pad.extract::<usize>() {
                (p_val, p_val)
            } else {
                pad.extract::<(usize, usize)>()?
            }
        } else {
            (0, 0)
        };

        let pool = MaxPool2d::new(k, s, p);
        Ok(PyMaxPool2d {
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
                "Input backend/dtype not supported for MaxPool2d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "MaxPool3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMaxPool3d {
    pub inner: MaxPool3d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyMaxPool3d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let k = if let Ok(s) = kernel_size.extract::<usize>() {
            (s, s, s)
        } else {
            kernel_size.extract::<(usize, usize, usize)>()?
        };

        let s = if let Some(st) = stride {
            if let Ok(s_val) = st.extract::<usize>() {
                Some((s_val, s_val, s_val))
            } else {
                Some(st.extract::<(usize, usize, usize)>()?)
            }
        } else {
            None
        };

        let p = if let Some(pad) = padding {
            if let Ok(p_val) = pad.extract::<usize>() {
                (p_val, p_val, p_val)
            } else {
                pad.extract::<(usize, usize, usize)>()?
            }
        } else {
            (0, 0, 0)
        };

        let pool = MaxPool3d::new(k, s, p);
        Ok(PyMaxPool3d {
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
                "Input backend/dtype not supported for MaxPool3d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}
