use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use ::tensor::Tensor;
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::rnn::{GRUCell, GRU as RustGRU};
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

#[derive(Clone)]
pub enum GRUWrapper {
    CpuF32(RustGRU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(RustGRU<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(RustGRU<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum GRUCellWrapper {
    CpuF32(GRUCell<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(GRUCell<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(coeus_nn::modules::rnn::GRUCell<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

/// PyGRU - Python wrapper for GRU layer
#[pyclass(name = "GRU", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyGRU {
    pub inner: GRUWrapper,
}

/// PyGRUCell - Python wrapper for GRUCell layer
#[pyclass(name = "GRUCell", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyGRUCell {
    pub inner: GRUCellWrapper,
}

#[pymethods]
impl PyGRUCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype="float32", device="cpu"))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        let dtype_str = dtype.unwrap_or("float32");
        let device_str = device.unwrap_or("cpu");

        let inner = match (device_str, dtype_str) {
            ("cpu", "float32") => {
                let cell = coeus_nn::modules::rnn::GRUCell::<
                    CpuBackend<Float32>,
                    DenseStorage<Float32>,
                    Float32,
                >::new(input_size, hidden_size, use_bias)
                .map_err(to_py_err)?;
                GRUCellWrapper::CpuF32(cell)
            }
            ("cpu", "float64") => {
                let cell = coeus_nn::modules::rnn::GRUCell::<
                    CpuBackend<Float64>,
                    DenseStorage<Float64>,
                    Float64,
                >::new(input_size, hidden_size, use_bias)
                .map_err(to_py_err)?;
                GRUCellWrapper::CpuF64(cell)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let cell = coeus_nn::modules::rnn::GRUCell::<
                    GpuBackend<Float32>,
                    DenseStorage<Float32>,
                    Float32,
                >::new(input_size, hidden_size, use_bias)
                .map_err(to_py_err)?;
                GRUCellWrapper::GpuF32(cell)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device_str, dtype_str
                )))
            }
        };
        Ok(PyGRUCell { inner })
    }

    fn __call__(&self, input: &PyTensor, hidden: Option<&PyTensor>) -> PyResult<PyTensor> {
        self.forward(input, hidden)
    }

    fn forward(&self, input: &PyTensor, hidden: Option<&PyTensor>) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (GRUCellWrapper::CpuF32(s), TensorWrapper::CpuDenseF32(i)) => {
                let h_opt = if let Some(h) = hidden {
                    if let TensorWrapper::CpuDenseF32(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Hidden state type mismatch",
                        ));
                    }
                } else {
                    None
                };

                let h = if let Some(h_val) = h_opt {
                    h_val.clone()
                } else {
                    Tensor::zeros_with_backend(
                        &[i.shape().dims()[0], s.hidden_size],
                        CpuBackend::default(),
                    )
                    .map_err(to_py_err)?
                };

                let output = s.forward_step(i, &h).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(output),
                })
            }
            (GRUCellWrapper::CpuF64(s), TensorWrapper::CpuDenseF64(i)) => {
                let h_opt = if let Some(h) = hidden {
                    if let TensorWrapper::CpuDenseF64(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Hidden state type mismatch",
                        ));
                    }
                } else {
                    None
                };

                let h = if let Some(h_val) = h_opt {
                    h_val.clone()
                } else {
                    Tensor::zeros_with_backend(
                        &[i.shape().dims()[0], s.hidden_size],
                        CpuBackend::default(),
                    )
                    .map_err(to_py_err)?
                };

                let output = s.forward_step(i, &h).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(output),
                })
            }
            #[cfg(feature = "gpu")]
            (GRUCellWrapper::GpuF32(s), TensorWrapper::GpuDenseF32(i)) => {
                let h_opt = if let Some(h) = hidden {
                    if let TensorWrapper::GpuDenseF32(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Hidden state type mismatch",
                        ));
                    }
                } else {
                    None
                };

                let h = if let Some(h_val) = h_opt {
                    h_val.clone()
                } else {
                    Tensor::zeros_generic_with_backend(
                        &[i.shape().dims()[0], s.hidden_size],
                        GpuBackend::default(),
                    )
                    .map_err(to_py_err)?
                };

                let output = s.forward_step(i, &h).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(output),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "GRUCell forward: device/dtype mismatch between container and input",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            GRUCellWrapper::CpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            GRUCellWrapper::CpuF64(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            GRUCellWrapper::GpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::GpuDenseF32(p.data().clone()),
                })
                .collect(),
        }
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            GRUCellWrapper::CpuF32(inner) => inner.train(mode),
            GRUCellWrapper::CpuF64(inner) => inner.train(mode),
            #[cfg(feature = "gpu")]
            GRUCellWrapper::GpuF32(inner) => inner.train(mode),
        }
    }
}

#[pymethods]
impl PyGRU {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false, dtype="float32", device="cpu"))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        bidirectional: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let layers = num_layers.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);
        let batch_first_val = batch_first.unwrap_or(false);
        let bidirectional_val = bidirectional.unwrap_or(false);
        let dtype_str = dtype.unwrap_or("float32");
        let device_str = device.unwrap_or("cpu");

        let inner = match (device_str, dtype_str) {
            ("cpu", "float32") => {
                let gru = RustGRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                GRUWrapper::CpuF32(gru)
            }
            ("cpu", "float64") => {
                let gru = RustGRU::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                GRUWrapper::CpuF64(gru)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") => {
                let gru = RustGRU::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                GRUWrapper::GpuF32(gru)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device_str, dtype_str
                )))
            }
        };
        Ok(PyGRU { inner })
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            GRUWrapper::CpuF32(inner) => inner.train(mode),
            GRUWrapper::CpuF64(inner) => inner.train(mode),
            #[cfg(feature = "gpu")]
            GRUWrapper::GpuF32(inner) => inner.train(mode),
        }
    }

    #[pyo3(signature = (input, hx=None))]
    fn __call__(&self, input: &PyTensor, hx: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        self.forward(input, hx)
    }

    #[pyo3(signature = (input, hx=None))]
    fn forward(&self, input: &PyTensor, hx: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        match (&self.inner, &input.inner) {
            (GRUWrapper::CpuF32(s), TensorWrapper::CpuDenseF32(i)) => {
                let h_val = if let Some(h) = hx {
                    if let TensorWrapper::CpuDenseF32(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "GRU hidden state dtype/device mismatch with module (expected CPU F32)",
                        ));
                    }
                } else {
                    None
                };
                let (output, hidden) = s.forward_with_hidden(i, h_val).map_err(to_py_err)?;
                Ok((
                    PyTensor {
                        inner: TensorWrapper::CpuDenseF32(output),
                    },
                    PyTensor {
                        inner: TensorWrapper::CpuDenseF32(hidden),
                    },
                ))
            }
            (GRUWrapper::CpuF64(s), TensorWrapper::CpuDenseF64(i)) => {
                let h_val = if let Some(h) = hx {
                    if let TensorWrapper::CpuDenseF64(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "GRU hidden state dtype/device mismatch with module (expected CPU F64)",
                        ));
                    }
                } else {
                    None
                };
                let (output, hidden) = s.forward_with_hidden(i, h_val).map_err(to_py_err)?;
                Ok((
                    PyTensor {
                        inner: TensorWrapper::CpuDenseF64(output),
                    },
                    PyTensor {
                        inner: TensorWrapper::CpuDenseF64(hidden),
                    },
                ))
            }
            #[cfg(feature = "gpu")]
            (GRUWrapper::GpuF32(s), TensorWrapper::GpuDenseF32(i)) => {
                let h_val = if let Some(h) = hx {
                    if let TensorWrapper::GpuDenseF32(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "GRU hidden state dtype/device mismatch with module (expected GPU F32)",
                        ));
                    }
                } else {
                    None
                };
                let (output, hidden) = s.forward_with_hidden(i, h_val).map_err(to_py_err)?;
                Ok((
                    PyTensor {
                        inner: TensorWrapper::GpuDenseF32(output),
                    },
                    PyTensor {
                        inner: TensorWrapper::GpuDenseF32(hidden),
                    },
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "GRU forward: device/dtype mismatch between container and input",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            GRUWrapper::CpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            GRUWrapper::CpuF64(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            GRUWrapper::GpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::GpuDenseF32(p.data().clone()),
                })
                .collect(),
        }
    }
}
