use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============ Sigmoid ============
#[pyclass(name = "Sigmoid", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySigmoid {}

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> Self {
        PySigmoid {}
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        input.sigmoid()
    }
}

// ============ Hardsigmoid ============
#[pyclass(name = "Hardsigmoid", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyHardsigmoid;

impl Default for PyHardsigmoid {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyHardsigmoid {
    #[new]
    pub fn new() -> Self {
        PyHardsigmoid
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)
        let dtype_str = match &input.inner {
            TensorWrapper::CpuDenseF32(_) => Some("float32"),
            TensorWrapper::CpuDenseF64(_) => Some("float64"),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => Some("float32"),
            _ => None,
        };
        let three = PyTensor::full(input.shape(), 3.0, dtype_str, None)?;
        let six = PyTensor::full(input.shape(), 6.0, dtype_str, None)?;
        let x_plus_3 = input.add(&three)?;
        let scaled = x_plus_3.div(&six)?;
        scaled.clamp(0.0, 1.0)
    }
}

// ============ LogSigmoid ============
#[pyclass(name = "LogSigmoid", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLogSigmoid;

#[pymethods]
impl PyLogSigmoid {
    #[new]
    fn new() -> Self {
        PyLogSigmoid
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // LogSigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x)))
        // For numerical stability: -softplus(-x) = -log(1 + exp(-x))
        // If x >= 0: -log(1 + exp(-x))
        // If x < 0: x - log(1 + exp(x))
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let data: Vec<Float32> = i
                    .as_slice()
                    .iter()
                    .map(|&x| {
                        let v = x.get();
                        let result = if v >= 0.0 {
                            -(1.0_f32 + (-v).exp()).ln()
                        } else {
                            v - (1.0_f32 + v.exp()).ln()
                        };
                        Float32::new(result)
                    })
                    .collect();
                let out = ::tensor::Tensor::from_vec(data, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(out),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let data: Vec<Float64> = i
                    .as_slice()
                    .iter()
                    .map(|&x| {
                        let v = x.get();
                        let result = if v >= 0.0 {
                            -(1.0_f64 + (-v).exp()).ln()
                        } else {
                            v - (1.0_f64 + v.exp()).ln()
                        };
                        Float64::new(result)
                    })
                    .collect();
                let out = ::tensor::Tensor::from_vec(data, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(out),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let data: Vec<Float32> = i
                    .as_slice()
                    .iter()
                    .map(|&x| {
                        let v = x.get();
                        let result = if v >= 0.0 {
                            -(1.0_f32 + (-v).exp()).ln()
                        } else {
                            v - (1.0_f32 + v.exp()).ln()
                        };
                        Float32::new(result)
                    })
                    .collect();
                let out = ::tensor::Tensor::from_vec(data, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(out),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type",
            )),
        }
    }
}

// ============ Softsign ============
#[pyclass(name = "Softsign", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySoftsign;

#[pymethods]
impl PySoftsign {
    #[new]
    fn new() -> Self {
        PySoftsign
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Softsign(x) = x / (1 + |x|)
        let abs_input = input.abs()?;
        let one_plus_abs = abs_input.add_scalar_f64(1.0)?;
        input.div(&one_plus_abs)
    }
}
