//! Python bindings for pooling layers.

use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use pyo3::types::PyAny;

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::pooling::{
    MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d,
};
use nn::core::module::Module;

/// PyMaxPool1d - Python wrapper for 1D max pooling
#[pyclass(name = "MaxPool1d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyMaxPool1d {
    pub inner: MaxPool1d,
}

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> PyResult<Self> {
        let p = padding.unwrap_or(0);
        let pool = MaxPool1d::new(kernel_size, stride, p);
        Ok(PyMaxPool1d { inner: pool })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// PyMaxPool2d - Python wrapper for 2D max pooling
#[pyclass(name = "MaxPool2d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyMaxPool2d {
    pub inner: MaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
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
        Ok(PyMaxPool2d { inner: pool })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// PyAvgPool1d - Python wrapper for 1D average pooling
#[pyclass(name = "AvgPool1d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyAvgPool1d {
    pub inner: AvgPool1d,
}

#[pymethods]
impl PyAvgPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> PyResult<Self> {
        let p = padding.unwrap_or(0);
        let pool = AvgPool1d::new(kernel_size, stride, p);
        Ok(PyAvgPool1d { inner: pool })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// PyAvgPool2d - Python wrapper for 2D average pooling
#[pyclass(name = "AvgPool2d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyAvgPool2d {
    pub inner: AvgPool2d,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: Bound<'_, PyAny>,
        stride: Option<Bound<'_, PyAny>>,
        padding: Option<Bound<'_, PyAny>>,
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

        let pool = AvgPool2d::new(k, s, p);
        Ok(PyAvgPool2d { inner: pool })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// PyAdaptiveAvgPool1d - Python wrapper for 1D adaptive average pooling
#[pyclass(name = "AdaptiveAvgPool1d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveAvgPool1d {
    pub inner: AdaptiveAvgPool1d,
}

#[pymethods]
impl PyAdaptiveAvgPool1d {
    #[new]
    fn new(output_size: usize) -> PyResult<Self> {
        let pool = AdaptiveAvgPool1d::new(output_size);
        Ok(PyAdaptiveAvgPool1d { inner: pool })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// PyAdaptiveAvgPool2d - Python wrapper for 2D adaptive average pooling
#[pyclass(name = "AdaptiveAvgPool2d", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyAdaptiveAvgPool2d {
    pub inner: AdaptiveAvgPool2d,
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    fn new(output_size: Bound<'_, PyAny>) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            (s, s)
        } else {
            output_size.extract::<(usize, usize)>()?
        };

        let pool = AdaptiveAvgPool2d::new(size);
        Ok(PyAdaptiveAvgPool2d { inner: pool })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMaxPool1d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyAvgPool1d>()?;
    m.add_class::<PyAvgPool2d>()?;
    m.add_class::<PyAdaptiveAvgPool1d>()?;
    m.add_class::<PyAdaptiveAvgPool2d>()?;
    Ok(())
}
