//! PyCoeus Softshrink and Hardshrink activation bindings

use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

// ============ Softshrink ============
/// Applies the soft shrinkage function element-wise:
/// Softshrink(x) = x - lambda if x > lambda
/// Softshrink(x) = x + lambda if x < -lambda
/// Softshrink(x) = 0 otherwise
#[pyclass(name = "Softshrink", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySoftshrink {
    pub lambd: f64,
}

#[pymethods]
impl PySoftshrink {
    #[new]
    #[pyo3(signature = (lambd=0.5))]
    fn new(lambd: f64) -> Self {
        PySoftshrink { lambd }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let data = i.as_slice();
                let lambd = Float32::new(self.lambd as f32);
                let neg_lambd = Float32::new(-self.lambd as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x > lambd {
                            Float32(x.0 - lambd.0)
                        } else if x < neg_lambd {
                            Float32(x.0 + lambd.0)
                        } else {
                            Float32(0.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(out) })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let data = i.as_slice();
                let lambd = Float64::new(self.lambd);
                let neg_lambd = Float64::new(-self.lambd);
                let result: Vec<Float64> = data
                    .iter()
                    .map(|&x| {
                        if x > lambd {
                            Float64(x.0 - lambd.0)
                        } else if x < neg_lambd {
                            Float64(x.0 + lambd.0)
                        } else {
                            Float64(0.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(out) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let data = i.as_slice();
                let lambd = Float32::new(self.lambd as f32);
                let neg_lambd = Float32::new(-self.lambd as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x > lambd {
                            Float32(x.0 - lambd.0)
                        } else if x < neg_lambd {
                            Float32(x.0 + lambd.0)
                        } else {
                            Float32(0.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(out) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for Softshrink",
            )),
        }
    }
}

// ============ Hardshrink ============
/// Applies the hard shrinkage function element-wise:
/// Hardshrink(x) = x if |x| > lambda
/// Hardshrink(x) = 0 otherwise
#[pyclass(name = "Hardshrink", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyHardshrink {
    pub lambd: f64,
}

#[pymethods]
impl PyHardshrink {
    #[new]
    #[pyo3(signature = (lambd=0.5))]
    fn new(lambd: f64) -> Self {
        PyHardshrink { lambd }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let data = i.as_slice();
                let lambd = Float32::new(self.lambd as f32);
                let neg_lambd = Float32::new(-self.lambd as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x > lambd || x < neg_lambd {
                            x
                        } else {
                            Float32(0.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(out) })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let data = i.as_slice();
                let lambd = Float64::new(self.lambd);
                let neg_lambd = Float64::new(-self.lambd);
                let result: Vec<Float64> = data
                    .iter()
                    .map(|&x| {
                        if x > lambd || x < neg_lambd {
                            x
                        } else {
                            Float64(0.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(out) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let data = i.as_slice();
                let lambd = Float32::new(self.lambd as f32);
                let neg_lambd = Float32::new(-self.lambd as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x > lambd || x < neg_lambd {
                            x
                        } else {
                            Float32(0.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(out) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for Hardshrink",
            )),
        }
    }
}
