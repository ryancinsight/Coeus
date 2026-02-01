use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::pooling::{AvgPool1d, AvgPool2d, AvgPool3d};
use pyo3::prelude::*;

#[pyclass(name = "AvgPool1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAvgPool1d {
    pub inner: AvgPool1d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAvgPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, ceil_mode=false, count_include_pad=true, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let p = padding.unwrap_or(0);
        let pool = AvgPool1d::new(kernel_size, stride, p, count_include_pad, ceil_mode);
        Ok(PyAvgPool1d {
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
                "Input backend/dtype not supported for AvgPool1d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AvgPool2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAvgPool2d {
    pub inner: AvgPool2d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, ceil_mode=false, count_include_pad=true, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
        ceil_mode: bool,
        count_include_pad: bool,
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

        let pool = AvgPool2d::new(k, s, p, count_include_pad, ceil_mode);
        Ok(PyAvgPool2d {
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
                "Input backend/dtype not supported for AvgPool2d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "AvgPool3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAvgPool3d {
    pub inner: AvgPool3d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyAvgPool3d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, ceil_mode=false, count_include_pad=true, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
        ceil_mode: bool,
        count_include_pad: bool,
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

        let pool = AvgPool3d::new(k, s, p, count_include_pad, ceil_mode);
        Ok(PyAvgPool3d {
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
                "Input backend/dtype not supported for AvgPool3d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}
