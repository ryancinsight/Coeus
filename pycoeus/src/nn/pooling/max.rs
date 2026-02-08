use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::pooling::{MaxPool1d, MaxPool2d, MaxPool3d};
use pyo3::prelude::*;
use backend::CpuBackend;
use storage::DenseStorage;
use dtype::float::{Float32, Float64};

#[derive(Clone)]
pub enum MaxPool1dWrapper {
    CpuF32(MaxPool1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(MaxPool1d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(MaxPool1d<backend::WgpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "MaxPool1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMaxPool1d {
    pub inner: MaxPool1dWrapper,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dilation=None, ceil_mode=None, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        ceil_mode: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let p = padding.unwrap_or(0);
        let dt = dtype.unwrap_or("float32");
        let dev = device.unwrap_or("cpu");
        let dil = dilation; // Passed as Option implicitly
        let cm = ceil_mode.unwrap_or(false);

        let wrapper = match (dev, dt) {
             ("cpu", "float32") => {
                 let m = MaxPool1d::new(kernel_size, stride, p, dil, cm);
                 MaxPool1dWrapper::CpuF32(m)
             }
             ("cpu", "float64") => {
                 let m = MaxPool1d::new(kernel_size, stride, p, dil, cm);
                 MaxPool1dWrapper::CpuF64(m)
             }
             #[cfg(feature = "gpu")]
             ("cuda" | "gpu", "float32") => {
                 let m = MaxPool1d::new(kernel_size, stride, p, dil, cm);
                 MaxPool1dWrapper::GpuF32(m)
             }
             _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                 format!("Unsupported device/dtype combination: {}/{}", dev, dt)
             )),
        };

        Ok(PyMaxPool1d {
            inner: wrapper,
            dtype: dt.to_string(),
            device: dev.to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (MaxPool1dWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (MaxPool1dWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            (MaxPool1dWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype mismatch with MaxPool1d module configuration",
            )),
        }
    }
}


#[derive(Clone)]
pub enum MaxPool2dWrapper {
    CpuF32(MaxPool2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(MaxPool2d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(MaxPool2d<backend::WgpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum MaxPool3dWrapper {
    CpuF32(MaxPool3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(MaxPool3d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(MaxPool3d<backend::WgpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "MaxPool2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyMaxPool2d {
    pub inner: MaxPool2dWrapper,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dilation=None, ceil_mode=None, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
        dilation: Option<Bound<'_, PyAny>>,
        ceil_mode: Option<bool>,
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

        let d = if let Some(dil) = dilation {
             if let Ok(d_val) = dil.extract::<usize>() {
                Some((d_val, d_val))
            } else {
                Some(dil.extract::<(usize, usize)>()?)
            }
        } else {
            None
        };

        let cm = ceil_mode.unwrap_or(false);
        let dt = dtype.unwrap_or("float32");
        let dev = device.unwrap_or("cpu");

        let wrapper = match (dev, dt) {
             ("cpu", "float32") => {
                 let m = MaxPool2d::new(k, s, p, d, cm);
                 MaxPool2dWrapper::CpuF32(m)
             }
             ("cpu", "float64") => {
                 let m = MaxPool2d::new(k, s, p, d, cm);
                 MaxPool2dWrapper::CpuF64(m)
             }
             #[cfg(feature = "gpu")]
             ("cuda" | "gpu", "float32") => {
                 let m = MaxPool2d::new(k, s, p, d, cm);
                 MaxPool2dWrapper::GpuF32(m)
             }
             _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                 format!("Unsupported device/dtype combination: {}/{}", dev, dt)
             )),
        };

        Ok(PyMaxPool2d {
            inner: wrapper,
            dtype: dt.to_string(),
            device: dev.to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (MaxPool2dWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (MaxPool2dWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            (MaxPool2dWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype mismatch with MaxPool2d module configuration",
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
    pub inner: MaxPool3dWrapper,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyMaxPool3d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, dilation=None, ceil_mode=None, dtype="float32", device="cpu"))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
        dilation: Option<Bound<'_, PyAny>>,
        ceil_mode: Option<bool>,
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

        let d = if let Some(dil) = dilation {
             if let Ok(d_val) = dil.extract::<usize>() {
                Some((d_val, d_val, d_val))
            } else {
                Some(dil.extract::<(usize, usize, usize)>()?)
            }
        } else {
            None
        };

        let cm = ceil_mode.unwrap_or(false);
        let dt = dtype.unwrap_or("float32");
        let dev = device.unwrap_or("cpu");

        let wrapper = match (dev, dt) {
             ("cpu", "float32") => {
                 let m = MaxPool3d::new(k, s, p, d, cm);
                 MaxPool3dWrapper::CpuF32(m)
             }
             ("cpu", "float64") => {
                 let m = MaxPool3d::new(k, s, p, d, cm);
                 MaxPool3dWrapper::CpuF64(m)
             }
             #[cfg(feature = "gpu")]
             ("cuda" | "gpu", "float32") => {
                 let m = MaxPool3d::new(k, s, p, d, cm);
                 MaxPool3dWrapper::GpuF32(m)
             }
             _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                 format!("Unsupported device/dtype combination: {}/{}", dev, dt)
             )),
        };

        Ok(PyMaxPool3d {
            inner: wrapper,
            dtype: dt.to_string(),
            device: dev.to_string(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (MaxPool3dWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (MaxPool3dWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            (MaxPool3dWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input backend/dtype mismatch with MaxPool3d module configuration",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}
