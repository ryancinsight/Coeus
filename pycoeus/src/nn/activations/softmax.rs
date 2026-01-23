use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::activation::Softplus;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============ Softmax ============
#[pyclass(name = "Softmax", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySoftmax {
    pub dim: isize,
}

#[pymethods]
impl PySoftmax {
    #[new]
    #[pyo3(signature = (dim=-1))]
    fn new(dim: isize) -> Self {
        PySoftmax { dim }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = coeus_nn::functional_api::softmax_dim(i, self.dim).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = coeus_nn::functional_api::softmax_dim(i, self.dim).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = coeus_nn::functional_api::softmax_dim(i, self.dim).map_err(to_py_err)?;
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

// ============ LogSoftmax ============
#[pyclass(name = "LogSoftmax", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLogSoftmax {
    pub dim: isize,
}

#[pymethods]
impl PyLogSoftmax {
    #[new]
    #[pyo3(signature = (dim=-1))]
    fn new(dim: isize) -> Self {
        PyLogSoftmax { dim }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = coeus_nn::functional_api::softmax_dim(i, self.dim).map_err(to_py_err)?;
                let res_log = ::tensor::ops::log(&res).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res_log),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = coeus_nn::functional_api::softmax_dim(i, self.dim).map_err(to_py_err)?;
                let res_log = ::tensor::ops::log(&res).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res_log),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = coeus_nn::functional_api::softmax_dim(i, self.dim).map_err(to_py_err)?;
                let res_log = coeus_nn::functional_api::log(&res).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res_log),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type",
            )),
        }
    }
}

// ============ Softmin ============
#[pyclass(name = "Softmin", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySoftmin {
    pub dim: isize,
}

#[pymethods]
impl PySoftmin {
    #[new]
    #[pyo3(signature = (dim=-1))]
    fn new(dim: isize) -> Self {
        PySoftmin { dim }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Softmin(x) = Softmax(-x)
        let neg_input = input.neg()?;
        let softmax = PySoftmax::new(self.dim);
        softmax.forward(&neg_input)
    }
}

// ============ Softplus ============
#[pyclass(name = "Softplus", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySoftplus {
    pub beta: f64,
    pub threshold: f64,
}

#[pymethods]
impl PySoftplus {
    #[new]
    #[pyo3(signature = (beta=1.0, threshold=20.0))]
    fn new(beta: f64, threshold: f64) -> Self {
        PySoftplus { beta, threshold }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = Softplus::new(
                    Float32::new(self.beta as f32),
                    Float32::new(self.threshold as f32),
                )
                .forward(i)
                .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = Softplus::new(Float64::new(self.beta), Float64::new(self.threshold))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = Softplus::new(
                    Float32::new(self.beta as f32),
                    Float32::new(self.threshold as f32),
                )
                .forward(i)
                .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for Softplus",
            )),
        }
    }
}
