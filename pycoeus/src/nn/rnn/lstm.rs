use super::to_py_err;
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::rnn::LSTM as RustLSTM;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

#[derive(Clone)]
pub enum LSTMWrapper {
    CpuF32(RustLSTM<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(RustLSTM<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(RustLSTM<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

/// PyLSTM - Python wrapper for LSTM layer
#[pyclass(name = "LSTM", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLSTM {
    pub inner: LSTMWrapper,
}

#[pymethods]
impl PyLSTM {
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
                let lstm = RustLSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                LSTMWrapper::CpuF32(lstm)
            }
            ("cpu", "float64") => {
                let lstm = RustLSTM::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                LSTMWrapper::CpuF64(lstm)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") => {
                let lstm = RustLSTM::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    input_size,
                    hidden_size,
                    layers,
                    use_bias,
                    batch_first_val,
                    bidirectional_val,
                )
                .map_err(to_py_err)?;
                LSTMWrapper::GpuF32(lstm)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype combination: {}/{}",
                    device_str, dtype_str
                )))
            }
        };
        Ok(PyLSTM { inner })
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            LSTMWrapper::CpuF32(inner) => inner.train(mode),
            LSTMWrapper::CpuF64(inner) => inner.train(mode),
            #[cfg(feature = "gpu")]
            LSTMWrapper::GpuF32(inner) => inner.train(mode),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        match (&self.inner, &input.inner) {
            (LSTMWrapper::CpuF32(s), TensorWrapper::CpuDenseF32(i)) => {
                let (output, (hn, cn)) = s.forward(i, None).map_err(to_py_err)?;
                Ok((
                    PyTensor {
                        inner: TensorWrapper::CpuDenseF32(output),
                    },
                    (
                        PyTensor {
                            inner: TensorWrapper::CpuDenseF32(hn),
                        },
                        PyTensor {
                            inner: TensorWrapper::CpuDenseF32(cn),
                        },
                    ),
                ))
            }
            (LSTMWrapper::CpuF64(s), TensorWrapper::CpuDenseF64(i)) => {
                let (output, (hn, cn)) = s.forward(i, None).map_err(to_py_err)?;
                Ok((
                    PyTensor {
                        inner: TensorWrapper::CpuDenseF64(output),
                    },
                    (
                        PyTensor {
                            inner: TensorWrapper::CpuDenseF64(hn),
                        },
                        PyTensor {
                            inner: TensorWrapper::CpuDenseF64(cn),
                        },
                    ),
                ))
            }
            #[cfg(feature = "gpu")]
            (LSTMWrapper::GpuF32(s), TensorWrapper::GpuDenseF32(i)) => {
                let (output, (hn, cn)) = s.forward(i, None).map_err(to_py_err)?;
                Ok((
                    PyTensor {
                        inner: TensorWrapper::GpuDenseF32(output),
                    },
                    (
                        PyTensor {
                            inner: TensorWrapper::GpuDenseF32(hn),
                        },
                        PyTensor {
                            inner: TensorWrapper::GpuDenseF32(cn),
                        },
                    ),
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "LSTM forward: device/dtype mismatch between container and input",
            )),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        match &self.inner {
            LSTMWrapper::CpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF32(p.data().clone()),
                })
                .collect(),
            LSTMWrapper::CpuF64(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::CpuDenseF64(p.data().clone()),
                })
                .collect(),
            #[cfg(feature = "gpu")]
            LSTMWrapper::GpuF32(s) => s
                .parameters()
                .into_iter()
                .map(|p| PyTensor {
                    inner: TensorWrapper::GpuDenseF32(p.data().clone()),
                })
                .collect(),
        }
    }
}
