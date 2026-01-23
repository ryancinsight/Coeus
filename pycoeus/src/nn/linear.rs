use crate::tensor::{PyTensor, TensorWrapper};
use pyo3::prelude::*;

use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::linear::LazyLinear;
use coeus_nn::modules::linear::Linear;
use dtype::float::{Float32, Float64};
use storage::DenseStorage;

#[derive(Clone)]
pub enum LinearWrapper {
    CpuF32(Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Linear<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Linear<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum LazyLinearWrapper {
    CpuF32(LazyLinear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyLinear<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyLinear<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Linear error: {}", e))
}

#[pyclass(name = "Linear", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLinear {
    pub inner: LinearWrapper,
    pub use_bias: bool,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true, dtype="float32", device="cpu"))]
    fn new(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in_features,
                    out_features,
                )
                .map_err(to_py_err)?;
                LinearWrapper::CpuF32(linear)
            }
            ("cpu", "float64") => {
                let linear = Linear::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    in_features,
                    out_features,
                )
                .map_err(to_py_err)?;
                LinearWrapper::CpuF64(linear)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let linear = Linear::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in_features,
                    out_features,
                )
                .map_err(to_py_err)?;
                LinearWrapper::GpuF32(linear)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device, dtype
                )))
            }
        };

        Ok(PyLinear {
            inner: wrapper,
            use_bias,
        })
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            LinearWrapper::CpuF32(m) => m.train(mode),
            LinearWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            LinearWrapper::GpuF32(m) => m.train(mode),
        }
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            LinearWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight.data().clone()),
            },
            LinearWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight.data().clone()),
            },
            #[cfg(feature = "gpu")]
            LinearWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight.data().clone()),
            },
        }
    }

    #[getter]
    fn bias(&self) -> Option<PyTensor> {
        if !self.use_bias {
            return None;
        }
        match &self.inner {
            LinearWrapper::CpuF32(m) => Some(PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.bias.data().clone()),
            }),
            LinearWrapper::CpuF64(m) => Some(PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.bias.data().clone()),
            }),
            #[cfg(feature = "gpu")]
            LinearWrapper::GpuF32(m) => Some(PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.bias.data().clone()),
            }),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LinearWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LinearWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LinearWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input/Module backend/dtype mismatch",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        params.push(self.weight());
        if let Some(b) = self.bias() {
            params.push(b);
        }
        params
    }
}

#[pyclass(name = "LazyLinear", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyLinear {
    pub inner: LazyLinearWrapper,
    pub use_bias: bool,
}

#[pymethods]
impl PyLazyLinear {
    #[new]
    #[pyo3(signature = (out_features, bias=true, dtype="float32", device="cpu"))]
    fn new(
        out_features: usize,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let linear = LazyLinear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_features,
                    use_bias,
                );
                LazyLinearWrapper::CpuF32(linear)
            }
            ("cpu", "float64") => {
                let linear = LazyLinear::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    out_features,
                    use_bias,
                );
                LazyLinearWrapper::CpuF64(linear)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let linear = LazyLinear::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    out_features,
                    use_bias,
                );
                LazyLinearWrapper::GpuF32(linear)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device, dtype
                )))
            }
        };

        Ok(PyLazyLinear {
            inner: wrapper,
            use_bias,
        })
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            LazyLinearWrapper::CpuF32(m) => m.train(mode),
            LazyLinearWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            LazyLinearWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyLinearWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyLinearWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyLinearWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input/Module backend/dtype mismatch",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            LazyLinearWrapper::CpuF32(m) => m
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            LazyLinearWrapper::CpuF64(m) => m
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            LazyLinearWrapper::GpuF32(m) => m
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::GpuDenseF32(p.data().clone()),
                })
                .collect(),
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinear>()?;
    m.add_class::<PyLazyLinear>()?;

    // Add to module __dict__ for dir() visibility (PyTorch compatibility)
    let dict = m.dict();
    dict.set_item("Linear", m.getattr("Linear")?)?;
    dict.set_item("LazyLinear", m.getattr("LazyLinear")?)?;

    Ok(())
}
