//! PyCoeus GLU and RReLU activation bindings

use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;

// ============ GLU (Gated Linear Unit) ============
/// Applies the gated linear unit function where the input is split in half
/// along the specified dimension, and gate = sigmoid(second_half):
/// GLU(x) = first_half * sigmoid(second_half)
#[pyclass(name = "GLU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyGLU {
    pub dim: i64,
}

#[pymethods]
impl PyGLU {
    #[new]
    #[pyo3(signature = (dim=-1))]
    fn new(dim: i64) -> Self {
        PyGLU { dim }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let shape = i.shape().dims();
                let ndim = shape.len();
                let dim = if self.dim < 0 {
                    (ndim as i64 + self.dim) as usize
                } else {
                    self.dim as usize
                };

                if dim >= ndim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Dimension out of range for GLU",
                    ));
                }

                let split_size = shape[dim];
                if split_size % 2 != 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "GLU requires even split size along dimension",
                    ));
                }

                // For last dimension, simple element-wise operation
                let data = i.as_slice();
                let half = split_size / 2;
                let total = data.len();

                // Calculate stride for the dimension
                let stride: usize = shape[dim + 1..].iter().product();
                let outer: usize = shape[..dim].iter().product();

                let mut result = Vec::with_capacity(total / 2);

                for outer_idx in 0..outer {
                    for inner_idx in 0..stride {
                        for h in 0..half {
                            let first_idx = outer_idx * split_size * stride + h * stride + inner_idx;
                            let second_idx = outer_idx * split_size * stride + (h + half) * stride + inner_idx;
                            let a = data[first_idx];
                            let b = data[second_idx];
                            // sigmoid(b) = 1 / (1 + exp(-b))
                            let sigmoid_b = Float32(1.0 / (1.0 + (-b.0).exp()));
                            result.push(Float32(a.0 * sigmoid_b.0));
                        }
                    }
                }

                let mut new_shape = shape.to_vec();
                new_shape[dim] = half;
                let out = tensor::Tensor::from_vec(result, &new_shape).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(out) })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let shape = i.shape().dims();
                let ndim = shape.len();
                let dim = if self.dim < 0 {
                    (ndim as i64 + self.dim) as usize
                } else {
                    self.dim as usize
                };

                if dim >= ndim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Dimension out of range for GLU",
                    ));
                }

                let split_size = shape[dim];
                if split_size % 2 != 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "GLU requires even split size along dimension",
                    ));
                }

                let data = i.as_slice();
                let half = split_size / 2;
                let total = data.len();

                let stride: usize = shape[dim + 1..].iter().product();
                let outer: usize = shape[..dim].iter().product();

                let mut result = Vec::with_capacity(total / 2);

                for outer_idx in 0..outer {
                    for inner_idx in 0..stride {
                        for h in 0..half {
                            let first_idx = outer_idx * split_size * stride + h * stride + inner_idx;
                            let second_idx = outer_idx * split_size * stride + (h + half) * stride + inner_idx;
                            let a = data[first_idx];
                            let b = data[second_idx];
                            let sigmoid_b = Float64(1.0 / (1.0 + (-b.0).exp()));
                            result.push(Float64(a.0 * sigmoid_b.0));
                        }
                    }
                }

                let mut new_shape = shape.to_vec();
                new_shape[dim] = half;
                let out = tensor::Tensor::from_vec(result, &new_shape).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(out) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for GLU",
            )),
        }
    }
}

// ============ RReLU (Randomized Leaky ReLU) ============
/// Applies the randomized leaky rectified linear unit function element-wise:
/// RReLU(x) = x if x >= 0
/// RReLU(x) = a * x if x < 0
/// where a is uniformly sampled from [lower, upper] during training,
/// and a = (lower + upper) / 2 during evaluation
#[pyclass(name = "RReLU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyRReLU {
    pub lower: f64,
    pub upper: f64,
    // Note: In evaluation mode, we use the mean of the range
}

#[pymethods]
impl PyRReLU {
    #[new]
    #[pyo3(signature = (lower=0.125, upper=0.3333333333333333))]
    fn new(lower: f64, upper: f64) -> Self {
        PyRReLU { lower, upper }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // For inference, use the average slope
        let slope = (self.lower + self.upper) / 2.0;

        match &input.inner {
            TensorWrapper::CpuDenseF32(i) => {
                let data = i.as_slice();
                let slope_f32 = Float32::new(slope as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x >= Float32::new(0.0) {
                            x
                        } else {
                            Float32(x.0 * slope_f32.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(out) })
            }
            TensorWrapper::CpuDenseF64(i) => {
                let data = i.as_slice();
                let slope_f64 = Float64::new(slope);
                let result: Vec<Float64> = data
                    .iter()
                    .map(|&x| {
                        if x >= Float64::new(0.0) {
                            x
                        } else {
                            Float64(x.0 * slope_f64.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(out) })
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(i) => {
                let data = i.as_slice();
                let slope_f32 = Float32::new(slope as f32);
                let result: Vec<Float32> = data
                    .iter()
                    .map(|&x| {
                        if x >= Float32::new(0.0) {
                            x
                        } else {
                            Float32(x.0 * slope_f32.0)
                        }
                    })
                    .collect();
                let out = tensor::Tensor::from_vec(result, i.shape().dims()).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(out) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported tensor type for RReLU",
            )),
        }
    }
}
