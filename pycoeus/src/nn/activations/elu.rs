use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::core::module::Module;
use coeus_nn::modules::activation::ELU;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============ ELU ============
#[pyclass(name = "ELU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyELU {
    pub alpha: f64,
}

#[pymethods]
impl PyELU {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        PyELU { alpha }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let res = ELU::new(Float32::new(self.alpha as f32))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let res = ELU::new(Float64::new(self.alpha))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let res = ELU::new(Float32::new(self.alpha as f32))
                    .forward(i)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for ELU",
            )),
        }
    }
}

// ============ SELU ============
#[pyclass(name = "SELU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PySELU;

#[pymethods]
impl PySELU {
    #[new]
    fn new() -> Self {
        PySELU
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // SELU(x) = scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
        // alpha = 1.6732632423543772848170429916717, scale = 1.0507009873554804934193349852946
        let alpha = 1.673_263_242_354_377_2_f64;
        let scale = 1.050_700_987_355_480_5_f64;

        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let elu_res = ELU::new(Float32::new(alpha as f32))
                    .forward(i)
                    .map_err(to_py_err)?;
                let res = elu_res
                    .mul_scalar(Float32::new(scale as f32))
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let elu_res = ELU::new(Float64::new(alpha))
                    .forward(i)
                    .map_err(to_py_err)?;
                let res = elu_res.mul_scalar(Float64::new(scale)).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let elu_res = ELU::new(Float32::new(alpha as f32))
                    .forward(i)
                    .map_err(to_py_err)?;
                let res = elu_res
                    .mul_scalar(Float32::new(scale as f32))
                    .map_err(to_py_err)?;
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

// ============ CELU ============
#[pyclass(name = "CELU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyCELU {
    pub alpha: f64,
}

#[pymethods]
impl PyCELU {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        PyCELU { alpha }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let alpha = Float32::new(self.alpha as f32);
                let data = i.as_slice();
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x >= Float32::new(0.0) {
                            x
                        } else {
                            alpha
                                * (Float32::new((x.get() / alpha.get()).exp()) - Float32::new(1.0))
                        }
                    })
                    .collect();
                let out =
                    ::tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(out),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let alpha = Float64::new(self.alpha);
                let data = i.as_slice();
                let result: Vec<Float64> = data
                    .iter()
                    .map(|&x| {
                        if x >= Float64::new(0.0) {
                            x
                        } else {
                            alpha
                                * (Float64::new((x.get() / alpha.get()).exp()) - Float64::new(1.0))
                        }
                    })
                    .collect();
                let out =
                    ::tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(out),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let alpha = Float32::new(self.alpha as f32);
                let data = i.as_slice();
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x >= Float32::new(0.0) {
                            x
                        } else {
                            alpha
                                * (Float32::new((x.get() / alpha.get()).exp()) - Float32::new(1.0))
                        }
                    })
                    .collect();
                let out =
                    crate::tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
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
