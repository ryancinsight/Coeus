use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::rnn::RNN as RustRNN;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

#[derive(Clone)]
pub enum RNNWrapper {
    CpuF32(RustRNN<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(RustRNN<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(RustRNN<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "RNN", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyRNN {
    pub inner: RNNWrapper,
}

#[pymethods]
impl PyRNN {
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
                let rnn = RustRNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                RNNWrapper::CpuF32(rnn)
            }
            ("cpu", "float64") => {
                let rnn = RustRNN::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                RNNWrapper::CpuF64(rnn)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") => {
                let rnn = RustRNN::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                RNNWrapper::GpuF32(rnn)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device_str, dtype_str
                )))
            }
        };
        Ok(PyRNN { inner })
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            RNNWrapper::CpuF32(inner) => inner.train(mode),
            RNNWrapper::CpuF64(inner) => inner.train(mode),
            #[cfg(feature = "gpu")]
            RNNWrapper::GpuF32(inner) => inner.train(mode),
        }
    }

    #[pyo3(signature = (input, hx=None))]
    fn __call__(&self, input: &PyTensor, hx: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        self.forward(input, hx)
    }

    #[pyo3(signature = (input, hx=None))]
    fn forward(&self, input: &PyTensor, hx: Option<&PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        match (&self.inner, &input.inner) {
            (RNNWrapper::CpuF32(s), TensorWrapper::CpuDenseF32(i)) => {
                let h_val = if let Some(h) = hx {
                    if let TensorWrapper::CpuDenseF32(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "RNN hidden state dtype/device mismatch with module (expected CPU F32)",
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
            (RNNWrapper::CpuF64(s), TensorWrapper::CpuDenseF64(i)) => {
                let h_val = if let Some(h) = hx {
                    if let TensorWrapper::CpuDenseF64(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "RNN hidden state dtype/device mismatch with module (expected CPU F64)",
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
            (RNNWrapper::GpuF32(s), TensorWrapper::GpuDenseF32(i)) => {
                let h_val = if let Some(h) = hx {
                    if let TensorWrapper::GpuDenseF32(h_inner) = &h.inner {
                        Some(h_inner)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "RNN hidden state dtype/device mismatch with module (expected GPU F32)",
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
                "RNN forward: device/dtype mismatch between container and input",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            RNNWrapper::CpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            RNNWrapper::CpuF64(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            RNNWrapper::GpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::GpuDenseF32(p.data().clone()),
                })
                .collect(),
        }
    }
}
