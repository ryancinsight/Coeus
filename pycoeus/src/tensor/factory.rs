use super::class::{to_py_err, PyTensor, TensorWrapper};
use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};
use dtype::int::Int64;
use pyo3::prelude::*;
use storage::DenseStorage;
use tensor::tensor_core::Tensor;

// Static factory methods for PyTensor
// Note: These are exposed via #[pyfunction] wrappers below, not via #[pymethods]
// to avoid conflicting with the main #[pymethods] block in ops.rs
impl PyTensor {
    pub fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dtype = dtype.unwrap_or("float32");
        match dtype {
            "float32" | "f32" => {
                let t =
                    Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&shape)
                        .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(t),
                })
            }
            "float64" | "f64" => {
                let t =
                    Tensor::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::zeros(&shape)
                        .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(t),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    pub fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dtype = dtype.unwrap_or("float32");
        match dtype {
            "float32" | "f32" => {
                let t = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&shape)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(t),
                })
            }
            "float64" | "f64" => {
                let t = Tensor::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::ones(&shape)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(t),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    pub fn randn(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dtype = dtype.unwrap_or("float32");
        match dtype {
            "float32" | "f32" => {
                let t =
                    Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::randn(&shape)
                        .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(t),
                })
            }
            "float64" | "f64" => {
                let t =
                    Tensor::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::randn(&shape)
                        .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(t),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    pub fn rand(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyTensor> {
        let dtype = dtype.unwrap_or("float32");
        match dtype {
            "float32" | "f32" => {
                let t = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::rand(&shape)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(t),
                })
            }
            "float64" | "f64" => {
                let t = Tensor::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::rand(&shape)
                    .map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(t),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    pub fn randint(low: i64, high: i64, shape: Vec<usize>) -> PyResult<PyTensor> {
        let t = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::randint(
            low, high, &shape,
        )
        .map_err(to_py_err)?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(t),
        })
    }

    pub fn zeros_like(input: &PyTensor) -> PyResult<PyTensor> {
        let inner = match &input.inner {
            TensorWrapper::CpuDenseF32(t) => {
                TensorWrapper::CpuDenseF32(Tensor::zeros_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuDenseF64(t) => {
                TensorWrapper::CpuDenseF64(Tensor::zeros_like(t).map_err(to_py_err)?)
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                TensorWrapper::GpuDenseF32(Tensor::zeros_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuSparseF32(t) => {
                TensorWrapper::CpuSparseF32(Tensor::zeros_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuSparseF64(t) => {
                TensorWrapper::CpuSparseF64(Tensor::zeros_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuDenseI64(t) => {
                TensorWrapper::CpuDenseI64(Tensor::zeros_like(t).map_err(to_py_err)?)
            }
            _ => return Err(to_py_err("zeros_like not implemented for this storage/dtype")),
        };

        Ok(PyTensor { inner })
    }

    pub fn ones_like(input: &PyTensor) -> PyResult<PyTensor> {
        let inner = match &input.inner {
            TensorWrapper::CpuDenseF32(t) => {
                TensorWrapper::CpuDenseF32(Tensor::ones_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuDenseF64(t) => {
                TensorWrapper::CpuDenseF64(Tensor::ones_like(t).map_err(to_py_err)?)
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => {
                TensorWrapper::GpuDenseF32(Tensor::ones_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuSparseF32(t) => {
                TensorWrapper::CpuSparseF32(Tensor::ones_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuSparseF64(t) => {
                TensorWrapper::CpuSparseF64(Tensor::ones_like(t).map_err(to_py_err)?)
            }
            TensorWrapper::CpuDenseI64(t) => {
                TensorWrapper::CpuDenseI64(Tensor::ones_like(t).map_err(to_py_err)?)
            }
            _ => return Err(to_py_err("ones_like not implemented for this storage/dtype")),
        };

        Ok(PyTensor { inner })
    }

    pub fn full_like(input: &PyTensor, fill_value: f32) -> PyResult<PyTensor> {
        let inner = match &input.inner {
            TensorWrapper::CpuDenseF32(t) => TensorWrapper::CpuDenseF32(
                Tensor::full_like(t, Float32(fill_value)).map_err(to_py_err)?,
            ),
            TensorWrapper::CpuDenseF64(t) => TensorWrapper::CpuDenseF64(
                Tensor::full_like(t, Float64(fill_value as f64)).map_err(to_py_err)?,
            ),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(t) => TensorWrapper::GpuDenseF32(
                Tensor::full_like(t, Float32(fill_value)).map_err(to_py_err)?,
            ),
            TensorWrapper::CpuSparseF32(t) => TensorWrapper::CpuSparseF32(
                Tensor::full_like(t, Float32(fill_value)).map_err(to_py_err)?,
            ),
            TensorWrapper::CpuSparseF64(t) => TensorWrapper::CpuSparseF64(
                Tensor::full_like(t, Float64(fill_value as f64)).map_err(to_py_err)?,
            ),
            TensorWrapper::CpuDenseI64(t) => TensorWrapper::CpuDenseI64(
                Tensor::full_like(t, Int64(fill_value as i64)).map_err(to_py_err)?,
            ),
            _ => return Err(to_py_err("full_like not implemented for this storage/dtype")),
        };

        Ok(PyTensor { inner })
    }

    pub fn eye(n: usize, m: Option<usize>) -> PyResult<PyTensor> {
        let m = m.unwrap_or(n);
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::eye(n, m)
            .map_err(to_py_err)?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(tensor),
        })
    }

    pub fn arange(start: f32, end: Option<f32>, step: f32) -> PyResult<PyTensor> {
        let (actual_start, actual_end) = match end {
            Some(e) => (start, e),
            None => (0.0, start),
        };
        let tensor = Tensor::arange(Float32(actual_start), Float32(actual_end), Float32(step))
            .map_err(to_py_err)?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(tensor),
        })
    }

    pub fn linspace(start: f32, end: f32, steps: usize) -> PyResult<PyTensor> {
        let tensor = Tensor::linspace(Float32(start), Float32(end), steps).map_err(to_py_err)?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(tensor),
        })
    }

    pub fn full(
        shape: Vec<usize>,
        fill_value: f64,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<PyTensor> {
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let t = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::full(
                    &shape,
                    Float32::new(fill_value as f32),
                )
                .map_err(to_py_err)?;
                TensorWrapper::CpuDenseF32(t)
            }
            ("cpu", "float64") => {
                let t = Tensor::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::full(
                    &shape,
                    Float64::new(fill_value),
                )
                .map_err(to_py_err)?;
                TensorWrapper::CpuDenseF64(t)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let t = Tensor::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::full(
                    &shape,
                    Float32::new(fill_value as f32),
                )
                .map_err(to_py_err)?;
                TensorWrapper::GpuDenseF32(t)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device, dtype
                )))
            }
        };

        Ok(PyTensor { inner: wrapper })
    }

    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
        let t = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            data.into_iter().map(Float32).collect(),
            &shape
        ).map_err(to_py_err)?;
        Ok(PyTensor {
             inner: TensorWrapper::CpuDenseF32(t)
        })
    }

    pub fn logspace(start: f32, end: f32, steps: usize, base: f64) -> PyResult<PyTensor> {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::logspace(
            Float32(start), Float32(end), steps, Float32(base as f32)
        ).map_err(to_py_err)?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(tensor),
        })
    }
}

#[pyfunction(name = "zeros")]
pub fn tensor_zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::zeros(shape, None)
}

#[pyfunction(name = "ones")]
pub fn tensor_ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::ones(shape, None)
}

#[pyfunction(name = "randn")]
pub fn tensor_randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::randn(shape, None)
}

#[pyfunction(name = "rand")]
pub fn tensor_rand(shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::rand(shape, None)
}

#[pyfunction(name = "randint")]
pub fn tensor_randint(low: i64, high: i64, shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::randint(low, high, shape)
}

#[pyfunction(name = "zeros_like")]
pub fn tensor_zeros_like(input: &PyTensor) -> PyResult<PyTensor> {
    PyTensor::zeros_like(input)
}

#[pyfunction(name = "ones_like")]
pub fn tensor_ones_like(input: &PyTensor) -> PyResult<PyTensor> {
    PyTensor::ones_like(input)
}

#[pyfunction(name = "logspace")]
pub fn tensor_logspace(start: f32, end: f32, steps: usize, base: f64) -> PyResult<PyTensor> {
    PyTensor::logspace(start, end, steps, base)
}

#[pyfunction(name = "full_like")]
pub fn tensor_full_like(input: &PyTensor, fill_value: f32) -> PyResult<PyTensor> {
    PyTensor::full_like(input, fill_value)
}

#[pyfunction(name = "arange")]
pub fn tensor_arange(start: f32, end: Option<f32>, step: f32) -> PyResult<PyTensor> {
    PyTensor::arange(start, end, step)
}

#[pyfunction(name = "linspace")]
pub fn tensor_linspace(start: f32, end: f32, steps: usize) -> PyResult<PyTensor> {
    PyTensor::linspace(start, end, steps)
}

#[pyfunction(name = "eye")]
pub fn tensor_eye(n: usize, m: Option<usize>) -> PyResult<PyTensor> {
    PyTensor::eye(n, m)
}

#[pyfunction(name = "full")]
pub fn tensor_full(
    shape: Vec<usize>,
    fill_value: f64,
    dtype: Option<&str>,
    device: Option<&str>,
) -> PyResult<PyTensor> {
    PyTensor::full(shape, fill_value, dtype, device)
}

#[pyfunction(name = "from_data")]
pub fn tensor_from_data(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::from_data(data, shape)
}

use pyo3::{wrap_pyfunction, Bound, PyResult, Python};

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tensor_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_randn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_rand, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_randint, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_full_like, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_logspace, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_arange, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_linspace, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_eye, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_full, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_from_data, m)?)?;
    Ok(())
}
