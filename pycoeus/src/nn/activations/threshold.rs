use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

// ============ Threshold ============
#[pyclass(name = "Threshold", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyThreshold {
    pub threshold: f64,
    pub value: f64,
}

#[pymethods]
impl PyThreshold {
    #[new]
    #[pyo3(signature = (threshold, value))]
    fn new(threshold: f64, value: f64) -> Self {
        PyThreshold { threshold, value }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Threshold(x) = x if x > threshold else value
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let data = i.as_slice();
                let thresh = Float32::new(self.threshold as f32);
                let val = Float32::new(self.value as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| if x > thresh { x } else { val })
                    .collect();
                let out =
                    ::tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(out),
                })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let data = i.as_slice();
                let thresh = Float64::new(self.threshold);
                let val = Float64::new(self.value);
                let result: Vec<Float64> = data
                    .iter()
                    .map(|&x| if x > thresh { x } else { val })
                    .collect();
                let out =
                    ::tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(out),
                })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let data = i.as_slice();
                let thresh = Float32::new(self.threshold as f32);
                let val = Float32::new(self.value as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| if x > thresh { x } else { val })
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
