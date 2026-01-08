//! Python bindings for RNN, LSTM, and GRU layers.

use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::rnn::{RNN as RustRNN, LSTM as RustLSTM, GRU as RustGRU};
use nn::core::module::Module;

/// PyRNN - Python wrapper for vanilla RNN layer
#[pyclass(name = "RNN", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyRNN {
    pub inner: RustRNN<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyRNN {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let layers = num_layers.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);
        let batch_first_val = batch_first.unwrap_or(false);
        let bidirectional_val = bidirectional.unwrap_or(false);
        
        let rnn = RustRNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            input_size,
            hidden_size,
            layers,
            use_bias,
            batch_first_val,
            bidirectional_val,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create RNN layer: {:?}",
                e
            ))
        })?;
        Ok(PyRNN { inner: rnn })
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        // Use forward_with_hidden to get both output and hidden state
        let (output, hidden) = self.inner.forward_with_hidden(&input.inner, None)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        
        Ok((PyTensor { inner: output }, PyTensor { inner: hidden }))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

/// PyLSTM - Python wrapper for LSTM layer
#[pyclass(name = "LSTM", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyLSTM {
    pub inner: RustLSTM<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyLSTM {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let layers = num_layers.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);
        let batch_first_val = batch_first.unwrap_or(false);
        let bidirectional_val = bidirectional.unwrap_or(false);
        
        let lstm = RustLSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            input_size,
            hidden_size,
            layers,
            use_bias,
            batch_first_val,
            bidirectional_val,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create LSTM layer: {:?}",
                e
            ))
        })?;
        Ok(PyLSTM { inner: lstm })
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        // LSTM.forward returns (output, (h_n, c_n)) as a tuple
        let (output, (h_n, c_n)) = self.inner.forward(&input.inner, None)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        
        Ok((
            PyTensor { inner: output },
            (PyTensor { inner: h_n }, PyTensor { inner: c_n }),
        ))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

/// PyGRU - Python wrapper for GRU layer
#[pyclass(name = "GRU", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PyGRU {
    pub inner: RustGRU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyGRU {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let layers = num_layers.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);
        let batch_first_val = batch_first.unwrap_or(false);
        let bidirectional_val = bidirectional.unwrap_or(false);
        
        let gru = RustGRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            input_size,
            hidden_size,
            layers,
            use_bias,
            batch_first_val,
            bidirectional_val,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create GRU layer: {:?}",
                e
            ))
        })?;
        Ok(PyGRU { inner: gru })
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        // Use forward_with_hidden to get both output and hidden state
        let (output, hidden) = self.inner.forward_with_hidden(&input.inner, None)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        
        Ok((PyTensor { inner: output }, PyTensor { inner: hidden }))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRNN>()?;
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyGRU>()?;
    Ok(())
}
