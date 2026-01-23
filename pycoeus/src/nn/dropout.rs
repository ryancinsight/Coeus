use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

use coeus_nn::core::module::Module;
use coeus_nn::modules::regularization::dropout::{Dropout, Dropout2d, Dropout3d};

#[pyclass(name = "Dropout", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyDropout {
    pub inner: Dropout,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5, dtype="float32", device="cpu"))]
    fn new(p: Option<f64>, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let probability = p.unwrap_or(0.5);
        let dropout = Dropout::new(probability);
        Ok(PyDropout {
            inner: dropout,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    fn train(&mut self, mode: bool) {
        self.inner.train(mode);
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
                "Input backend/dtype not supported for Dropout",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "Dropout2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyDropout2d {
    pub inner: Dropout2d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyDropout2d {
    #[new]
    #[pyo3(signature = (p=0.5, dtype="float32", device="cpu"))]
    fn new(p: Option<f64>, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let probability = p.unwrap_or(0.5);
        let dropout = Dropout2d::new(probability);
        Ok(PyDropout2d {
            inner: dropout,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    fn train(&mut self, mode: bool) {
        self.inner.train(mode);
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
                "Input backend/dtype not supported for Dropout2d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

#[pyclass(name = "Dropout3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyDropout3d {
    pub inner: Dropout3d,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyDropout3d {
    #[new]
    #[pyo3(signature = (p=0.5, dtype="float32", device="cpu"))]
    fn new(p: Option<f64>, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let probability = p.unwrap_or(0.5);
        let dropout = Dropout3d::new(probability);
        Ok(PyDropout3d {
            inner: dropout,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    fn train(&mut self, mode: bool) {
        self.inner.train(mode);
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
                "Input backend/dtype not supported for Dropout3d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

// Dropout1d is an alias for Dropout in PyTorch for 1D inputs
#[pyclass(name = "Dropout1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyDropout1d {
    pub inner: Dropout,
    pub dtype: String,
    pub device: String,
}

#[pymethods]
impl PyDropout1d {
    #[new]
    #[pyo3(signature = (p=0.5, dtype="float32", device="cpu"))]
    fn new(p: Option<f64>, dtype: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let probability = p.unwrap_or(0.5);
        let dropout = Dropout::new(probability);
        Ok(PyDropout1d {
            inner: dropout,
            dtype: dtype.unwrap_or("float32").to_string(),
            device: device.unwrap_or("cpu").to_string(),
        })
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    fn train(&mut self, mode: bool) {
        self.inner.train(mode);
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
                "Input backend/dtype not supported for Dropout1d",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

// AlphaDropout for SELU networks
#[pyclass(name = "AlphaDropout", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyAlphaDropout {
    pub p: f64,
    pub training: bool,
}

#[pymethods]
impl PyAlphaDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: Option<f64>) -> Self {
        PyAlphaDropout {
            p: p.unwrap_or(0.5),
            training: true,
        }
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // AlphaDropout maintains the self-normalizing property
        // For simplicity, use standard dropout when training
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        // Simplified implementation: use regular dropout
        // Full AlphaDropout would need to maintain mean/variance
        let dropout = Dropout::new(self.p);
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = dropout.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = dropout.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = dropout.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDropout>()?;
    m.add_class::<PyDropout1d>()?;
    m.add_class::<PyDropout2d>()?;
    m.add_class::<PyDropout3d>()?;
    m.add_class::<PyAlphaDropout>()?;

    // Add to module __dict__ for dir() visibility (PyTorch compatibility)
    let dict = m.dict();
    dict.set_item("Dropout", m.getattr("Dropout")?)?;
    dict.set_item("Dropout1d", m.getattr("Dropout1d")?)?;
    dict.set_item("Dropout2d", m.getattr("Dropout2d")?)?;
    dict.set_item("Dropout3d", m.getattr("Dropout3d")?)?;
    dict.set_item("AlphaDropout", m.getattr("AlphaDropout")?)?;

    Ok(())
}
