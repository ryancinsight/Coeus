use crate::tensor::{PyTensor, TensorWrapper};
use pyo3::prelude::*;

use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::convolution::{LazyConv1d, LazyConv2d, LazyConv3d};
use dtype::float::{Float32, Float64};
use storage::DenseStorage;

use super::to_py_err;

#[derive(Clone)]
pub enum LazyConv1DWrapper {
    CpuF32(LazyConv1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyConv1d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyConv1d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum LazyConv2DWrapper {
    CpuF32(LazyConv2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyConv2d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyConv2d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum LazyConv3DWrapper {
    CpuF32(LazyConv3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyConv3d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyConv3d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "LazyConv1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyConv1d {
    inner: LazyConv1DWrapper,
}

#[pymethods]
impl PyLazyConv1d {
    #[new]
    #[pyo3(signature = (out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true, padding_mode="zeros", dtype="float32", device="cpu"))]
    fn new(
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
        padding_mode: Option<&str>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let bias = bias.unwrap_or(true);
        let _ = padding_mode;
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer = LazyConv1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                );
                LazyConv1DWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer = LazyConv1d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                );
                LazyConv1DWrapper::CpuF64(layer)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let layer = LazyConv1d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                );
                LazyConv1DWrapper::GpuF32(layer)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype: {}/{}",
                    device, dtype
                )))
            }
        };

        Ok(PyLazyConv1d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyConv1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyConv1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyConv1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Backend mismatch",
            )),
        }
    }
}

#[pyclass(name = "LazyConv2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyConv2d {
    inner: LazyConv2DWrapper,
}

#[pymethods]
impl PyLazyConv2d {
    #[new]
    #[pyo3(signature = (out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=1, bias=true, padding_mode="zeros", dtype="float32", device="cpu"))]
    fn new(
        out_channels: usize,
        kernel_size: &Bound<'_, PyAny>,
        stride: Option<&Bound<'_, PyAny>>,
        padding: Option<&Bound<'_, PyAny>>,
        dilation: Option<&Bound<'_, PyAny>>,
        groups: Option<usize>,
        bias: Option<bool>,
        padding_mode: Option<&str>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let k = if let Ok(v) = kernel_size.extract::<usize>() {
            (v, v)
        } else {
            kernel_size.extract::<(usize, usize)>()?
        };

        // Handling optionals slightly better
        let s = if let Some(v) = stride {
            if let Ok(i) = v.extract::<usize>() {
                (i, i)
            } else {
                v.extract::<(usize, usize)>()?
            }
        } else {
            (1, 1)
        };
        let p = if let Some(v) = padding {
            if let Ok(i) = v.extract::<usize>() {
                (i, i)
            } else {
                v.extract::<(usize, usize)>()?
            }
        } else {
            (0, 0)
        };
        let d = if let Some(v) = dilation {
            if let Ok(i) = v.extract::<usize>() {
                (i, i)
            } else {
                v.extract::<(usize, usize)>()?
            }
        } else {
            (1, 1)
        };

        let groups = groups.unwrap_or(1);
        let bias = bias.unwrap_or(true);
        let _ = padding_mode;
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer = LazyConv2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_channels,
                    k,
                    s,
                    p,
                    d,
                    groups,
                    bias,
                );
                LazyConv2DWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer = LazyConv2d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    out_channels,
                    k,
                    s,
                    p,
                    d,
                    groups,
                    bias,
                );
                LazyConv2DWrapper::CpuF64(layer)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let layer = LazyConv2d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_channels,
                    k,
                    s,
                    p,
                    d,
                    groups,
                    bias,
                );
                LazyConv2DWrapper::GpuF32(layer)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype: {}/{}",
                    device, dtype
                )))
            }
        };
        Ok(PyLazyConv2d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyConv2DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyConv2DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyConv2DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Backend mismatch",
            )),
        }
    }
}

#[pyclass(name = "LazyConv3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyConv3d {
    inner: LazyConv3DWrapper,
}

#[pymethods]
impl PyLazyConv3d {
    #[new]
    #[pyo3(signature = (out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=1, bias=true, padding_mode="zeros", dtype="float32", device="cpu"))]
    fn new(
        out_channels: usize,
        kernel_size: &Bound<'_, PyAny>,
        stride: Option<&Bound<'_, PyAny>>,
        padding: Option<&Bound<'_, PyAny>>,
        dilation: Option<&Bound<'_, PyAny>>,
        groups: Option<usize>,
        bias: Option<bool>,
        padding_mode: Option<&str>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let k = if let Ok(v) = kernel_size.extract::<usize>() {
            (v, v, v)
        } else {
            kernel_size.extract::<(usize, usize, usize)>()?
        };
        let s = if let Some(v) = stride {
            if let Ok(i) = v.extract::<usize>() {
                (i, i, i)
            } else {
                v.extract::<(usize, usize, usize)>()?
            }
        } else {
            (1, 1, 1)
        };
        let p = if let Some(v) = padding {
            if let Ok(i) = v.extract::<usize>() {
                (i, i, i)
            } else {
                v.extract::<(usize, usize, usize)>()?
            }
        } else {
            (0, 0, 0)
        };
        let d = if let Some(v) = dilation {
            if let Ok(i) = v.extract::<usize>() {
                (i, i, i)
            } else {
                v.extract::<(usize, usize, usize)>()?
            }
        } else {
            (1, 1, 1)
        };

        let groups = groups.unwrap_or(1);
        let bias = bias.unwrap_or(true);
        let _ = padding_mode;
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer = LazyConv3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_channels,
                    k,
                    s,
                    p,
                    d,
                    groups,
                    bias,
                );
                LazyConv3DWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer = LazyConv3d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    out_channels,
                    k,
                    s,
                    p,
                    d,
                    groups,
                    bias,
                );
                LazyConv3DWrapper::CpuF64(layer)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let layer = LazyConv3d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_channels,
                    k,
                    s,
                    p,
                    d,
                    groups,
                    bias,
                );
                LazyConv3DWrapper::GpuF32(layer)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype: {}/{}",
                    device, dtype
                )))
            }
        };
        Ok(PyLazyConv3d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyConv3DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyConv3DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyConv3DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Backend mismatch",
            )),
        }
    }
}
